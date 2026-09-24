import sys
import time
import math
import threading
import csv
from concurrent.futures import ThreadPoolExecutor

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

N_STEPS = 100_000_000
TRUE_PI = math.pi


if HAVE_NUMBA:
    @njit
    def calc_pi_serial(num_steps):
        step = 1.0 / num_steps
        total = 0.0
        for i in range(num_steps):
            x = (i + 0.5) * step
            total += 4.0 / (1.0 + x * x)
        return total * step
else:
    def calc_pi_serial(num_steps):
        step = 1.0 / num_steps
        total = 0.0
        for i in range(num_steps):
            x = (i + 0.5) * step
            total += 4.0 / (1.0 + x * x)
        return total * step


if HAVE_NUMBA:
    @njit(parallel=True)
    def calc_pi_reduction(num_steps):
        step = 1.0 / num_steps
        total = 0.0
        for i in prange(num_steps):
            x = (i + 0.5) * step
            total += 4.0 / (1.0 + x * x)
        return total * step


def calc_pi_reduction_manual(num_steps, num_threads):
    step = 1.0 / num_steps
    chunk = num_steps // num_threads

    def partial_sum(start, end):
        local_total = 0.0
        for i in range(start, end):
            x = (i + 0.5) * step
            local_total += 4.0 / (1.0 + x * x)
        return local_total

    ranges = []
    for t in range(num_threads):
        start = t * chunk
        end = num_steps if t == num_threads - 1 else start + chunk
        ranges.append((start, end))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(partial_sum, s, e) for s, e in ranges]
        partials = [f.result() for f in futures]

    return sum(partials) * step


def calc_pi_naive_race(num_steps, num_threads):
    step = 1.0 / num_steps
    chunk = num_steps // num_threads
    shared_sum = [0.0]

    def worker_full(start, end):
        for i in range(start, end):
            x = (i + 0.5) * step
            term = 4.0 / (1.0 + x * x)
            shared_sum[0] += term

    threads = []
    for t in range(num_threads):
        start = t * chunk
        end = num_steps if t == num_threads - 1 else start + chunk
        th = threading.Thread(target=worker_full, args=(start, end))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()

    return shared_sum[0] * step


def calc_pi_critical_section(num_steps, num_threads):
    step = 1.0 / num_steps
    chunk = num_steps // num_threads
    shared_sum = [0.0]
    lock = threading.Lock()

    def worker(start, end):
        for i in range(start, end):
            x = (i + 0.5) * step
            term = 4.0 / (1.0 + x * x)
            with lock:
                shared_sum[0] += term

    threads = []
    for t in range(num_threads):
        start = t * chunk
        end = num_steps if t == num_threads - 1 else start + chunk
        th = threading.Thread(target=worker, args=(start, end))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()

    return shared_sum[0] * step


def task_serial():
    if HAVE_NUMBA:
        calc_pi_serial(1000)
    t0 = time.perf_counter()
    pi = calc_pi_serial(N_STEPS)
    t1 = time.perf_counter()
    print(f"Serial: Pi = {pi:.12f} | Time = {t1-t0:.4f}s | Error = {abs(pi-TRUE_PI):.2e}")
    return t1 - t0


def task_2_1_race(outfile="task2_1_race.csv"):
    print("Task 2.1: Race Condition Quantification")
    print(f"{'P':>4} | {'Pi':>15} | {'Error':>12}")
    results = []
    n_small = 5_000_000
    for p in [1, 2, 4, 8]:
        pi = calc_pi_naive_race(n_small, p)
        err = abs(pi - TRUE_PI)
        print(f"{p:>4} | {pi:>15.8f} | {err:>12.2e}")
        results.append((p, pi, err))
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "pi_computed", "abs_error"])
        w.writerows(results)
    print(f"Saved to {outfile}")


def task_2_2_critical():
    print("Task 2.2: Critical Section Overhead")
    n = 1_000_000
    t0 = time.perf_counter()
    pi_serial = calc_pi_serial(n)
    t1 = time.perf_counter()
    serial_time = t1 - t0

    t2 = time.perf_counter()
    pi_crit = calc_pi_critical_section(n, 4)
    t3 = time.perf_counter()
    crit_time = t3 - t2

    overhead_pct = ((crit_time - serial_time) / serial_time) * 100
    print(f"Serial:          Pi = {pi_serial:.8f} | Time = {serial_time:.4f}s")
    print(f"Critical(P=4):   Pi = {pi_crit:.8f} | Time = {crit_time:.4f}s")
    print(f"Lock contention overhead: {overhead_pct:.1f}%")


def task_2_3_reduction(outfile="task2_3_reduction.csv"):
    print("Task 2.3: Strong Scaling Benchmark (Parallel Reduction)")
    if HAVE_NUMBA:
        calc_pi_reduction(1000)
    results = []
    for p in [1, 2, 4, 8, 16]:
        times = []
        for trial in range(5):
            t0 = time.perf_counter()
            pi = calc_pi_reduction_manual(N_STEPS, p)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_time = sum(times) / len(times)
        results.append((p, avg_time))
        print(f"P={p:>2} | avg time over 5 trials = {avg_time:.4f}s")
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "avg_time_seconds"])
        w.writerows(results)
    print(f"Saved to {outfile}")
    return results


def task_2_4_speedup(outfile="task2_4_speedup.csv"):
    print("Task 2.4: Speedup & Efficiency")
    results = task_2_3_reduction()
    t1 = dict(results)[1]
    rows = []
    print(f"{'P':>4} | {'T(P)':>10} | {'Speedup':>10} | {'Efficiency':>10}")
    for p, t in results:
        speedup = t1 / t
        efficiency = speedup / p
        rows.append((p, t, speedup, efficiency))
        print(f"{p:>4} | {t:>10.4f} | {speedup:>10.2f} | {efficiency:>10.2f}")
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "time_seconds", "speedup", "efficiency"])
        w.writerows(rows)
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "serial"
    if not HAVE_NUMBA:
        print("numba not installed, falling back to pure Python.")

    if mode == "serial":
        task_serial()
    elif mode == "race":
        task_2_1_race()
    elif mode == "critical":
        task_2_2_critical()
    elif mode == "reduction":
        task_2_3_reduction()
    elif mode == "speedup":
        task_2_4_speedup()