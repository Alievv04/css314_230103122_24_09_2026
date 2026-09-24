
import sys, time
import numpy as np
from numba import njit
from multiprocessing import Pool, Value, Barrier, shared_memory

N = 10_000_000 + 3122 * 1000      
MOD = 1_000_000_007
MAX_THREADS = 64
PAD = 16                           # 16 int32 = 64 байта = одна кэш-линия


# ---------- ядро (компилируется в машинный код) ----------
@njit(cache=True)
def k_reduce(lo, hi):              # диапазон [lo, hi)
    mx = 0; s = 0; h = 0
    for i in range(lo, hi):
        n = i; st = 0
        while n > 1:
            if n & 1 == 0:
                n >>= 1
            else:
                n = 3 * n + 1
            st += 1
        if st > mx: mx = st
        s += st
        if st > 100: h += 1
    return mx, s, h

@njit(cache=True)
def k_shared(lo, hi, arr, idx):    # счётчик hits пишется в общий массив
    mx = 0; s = 0
    for i in range(lo, hi):
        n = i; st = 0
        while n > 1:
            if n & 1 == 0:
                n >>= 1
            else:
                n = 3 * n + 1
            st += 1
        if st > mx: mx = st
        s += st
        if st > 100:
            arr[idx] += 1
    return mx, s


# ---------- код рабочих процессов ----------
WID = K = BARRIER = ARR = _SHM = None

def init(counter, barrier, shm_name, k):
    global WID, K, BARRIER, ARR, _SHM
    with counter.get_lock():
        WID = counter.value
        counter.value += 1
    K, BARRIER = k, barrier
    _SHM = shared_memory.SharedMemory(name=shm_name)
    ARR = np.ndarray((MAX_THREADS * PAD,), dtype=np.int32, buffer=_SHM.buf)
    k_reduce(1, 10); k_shared(1, 10, ARR, 0)      # прогрев JIT

def w_static(args):
    """schedule(static[,chunk]): работа закреплена за процессом заранее."""
    chunk, variant = args
    BARRIER.wait()                 # гарантирует: каждая задача уходит разному процессу
    if chunk == 0:
        base = N // K
        lo = 1 + WID * base
        hi = N + 1 if WID == K - 1 else lo + base
        ranges = [(lo, hi)]
    else:                          # round-robin по кускам
        ranges = [(s, min(s + chunk, N + 1))
                  for s in range(1 + WID * chunk, N + 1, K * chunk)]
    mx = s_tot = h_tot = 0
    for lo, hi in ranges:
        if variant == "reduce":
            m, s, h = k_reduce(lo, hi)
            h_tot += h
        else:
            idx = WID if variant == "naive" else WID * PAD
            m, s = k_shared(lo, hi, ARR, idx)
        mx = max(mx, m); s_tot += s
    return mx, s_tot, h_tot

def w_dyn(rng):
    return k_reduce(rng[0], rng[1])


# ---------- планировщики ----------
def dynamic_chunks(c):
    return [(s, min(s + c, N + 1)) for s in range(1, N + 1, c)]

def guided_chunks(k, min_chunk=100):
    out, lo = [], 1
    while lo <= N:
        size = max(min_chunk, (N - lo + 1) // k)
        out.append((lo, min(lo + size, N + 1)))
        lo += size
    return out


def main():
    k, mode = int(sys.argv[1]), sys.argv[2]

    if mode == "seq":
        k_reduce(1, 10)            # прогрев JIT
        pool = None
    else:
        counter = Value("i", 0)
        barrier = Barrier(k)
        shm = shared_memory.SharedMemory(create=True, size=MAX_THREADS * PAD * 4)
        arr = np.ndarray((MAX_THREADS * PAD,), dtype=np.int32, buffer=shm.buf)
        pool = Pool(k, initializer=init, initargs=(counter, barrier, shm.name, k))

    static_cfg = {"static": (0, "reduce"), "static1000": (1000, "reduce"),
                  "naive": (0, "naive"), "padded": (0, "padded")}
    chunks = {"dyn100": dynamic_chunks(100), "dyn10000": dynamic_chunks(10000),
              "guided": guided_chunks(k)}.get(mode)

    for run in range(1, 4):
        if pool: arr[:] = 0
        t0 = time.perf_counter()

        if mode == "seq":
            mx, s, h = k_reduce(1, N + 1)
        elif mode in static_cfg:
            res = pool.map(w_static, [static_cfg[mode]] * k, chunksize=1)
            mx = max(r[0] for r in res); s = sum(r[1] for r in res)
            if mode == "naive":    h = int(arr[:MAX_THREADS].sum())
            elif mode == "padded": h = int(arr[::PAD].sum())
            else:                  h = sum(r[2] for r in res)
        else:                      # dyn100 / dyn10000 / guided
            res = list(pool.imap_unordered(w_dyn, chunks, chunksize=1))
            mx = max(r[0] for r in res); s = sum(r[1] for r in res)
            h = sum(r[2] for r in res)

        t1 = time.perf_counter()
        print(f"{mode},{k},{run},{t1 - t0:.6f},{mx},{s % MOD},{h}")

    if pool:
        pool.close(); pool.join(); shm.close(); shm.unlink()

if __name__ == "__main__":         # обязательно на Windows
    main()