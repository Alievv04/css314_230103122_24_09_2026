
import sys, os, csv, subprocess
import matplotlib.pyplot as plt
from collatz import N as N_ITER          # тот же N, что в collatz.py (меняй только там)

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = [sys.executable, os.path.join(HERE, "collatz.py")]

LOGICAL = os.cpu_count()                 # логические потоки
MAX_PHYS = 8                             # <-- число ФИЗИЧЕСКИХ ядер твоего CPU (для экспериментов A и B)
KS = sorted({k for k in [1, 2, 4, 8, 16, LOGICAL] if k <= LOGICAL})   # k=2 обязателен для p


def run(k, mode):
    r = subprocess.run(EXE + [str(k), mode], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"collatz.py failed for k={k}, mode={mode}")
    rows = [l.split(",") for l in r.stdout.strip().splitlines()]
    times = [float(x[3]) for x in rows]              # [run1, run2, run3]
    check = (rows[0][4], rows[0][5], rows[0][6])     # max, checksum, hits
    return times, check


def avg23(t):
    return (t[1] + t[2]) / 2.0                       # Run 1 (cold) отбрасывается


all_rows = []

# ---------- Таблица 1: масштабирование ----------
print("Sequential baseline...")
t_seq_runs, ref = run(1, "seq")
T_seq = avg23(t_seq_runs)
all_rows.append(["table1", "seq", 1, *t_seq_runs, T_seq])
print(f"reference (max, checksum, hits) = {ref}")

Tk, Semp = {}, {}
for k in KS:
    print(f"static, k={k}...")
    t, chk = run(k, "static")
    assert chk == ref, f"MISMATCH at k={k}: {chk} vs {ref}"
    Tk[k] = avg23(t)
    Semp[k] = T_seq / Tk[k]
    all_rows.append(["table1", "static", k, *t, Tk[k]])

p = 2 * (1 - 1 / Semp[2])
Stheo = {k: 1 / ((1 - p) + p / k) for k in KS}
print(f"\nT_seq = {T_seq:.4f} s   p = {p:.4f}")
for k in KS:
    print(f"k={k:2d}  T={Tk[k]:.4f}  S_emp={Semp[k]:.3f}  "
          f"S_theo={Stheo[k]:.3f}  Delta={Stheo[k] - Semp[k]:.3f}")

# ---------- Эксперимент A: false sharing ----------
print("\nExperiment A...")
tA = {}
for mode in ["naive", "padded"]:
    t, chk = run(MAX_PHYS, mode)
    assert chk == ref, f"MISMATCH in {mode}: {chk} vs {ref}"
    tA[mode] = avg23(t)
    all_rows.append(["expA", mode, MAX_PHYS, *t, tA[mode]])
    print(f"{mode}: T={tA[mode]:.4f} s   throughput={N_ITER / tA[mode]:.3e} iter/s")
print("Penalty ratio naive/padded =", tA["naive"] / tA["padded"])

# ---------- Эксперимент B: планировщики ----------
print("\nExperiment B...")
for mode in ["static", "static1000", "dyn100", "dyn10000", "guided"]:
    t, chk = run(MAX_PHYS, mode)
    assert chk == ref, f"MISMATCH in {mode}: {chk} vs {ref}"
    all_rows.append(["expB", mode, MAX_PHYS, *t, avg23(t)])
    print(f"{mode}: T={avg23(t):.4f} s")

# ---------- results.csv ----------
with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table", "mode", "threads", "run1_cold", "run2", "run3", "avg_s"])
    w.writerows(all_rows)

# ---------- график ----------
plt.figure(figsize=(7, 5))
plt.plot(KS, KS, "--", label="Linear ideal")
plt.plot(KS, [Stheo[k] for k in KS], "o-", label=f"S_theo (Amdahl, p={p:.3f})")
plt.plot(KS, [Semp[k] for k in KS], "s-", label="S_emp")
plt.xlabel("Threads k")
plt.ylabel("Speedup")
plt.grid(True)
plt.legend()
plt.savefig("speedup_plot.png", dpi=300)
print("\nSaved results.csv and speedup_plot.png")