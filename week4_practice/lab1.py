import sys
import time
import threading
import math
from concurrent.futures import ThreadPoolExecutor
import csv


def worker_task(thread_id: int, team_size: int, log_lines: list):
    native_tid = threading.get_native_id()
    role = "Master" if thread_id == 0 else "Worker"
    time.sleep(0.001 * (thread_id % 3))
    line = f"[{role}] Logical Rank: {thread_id} of {team_size} | Native OS TID: {native_tid}"
    print(line)
    log_lines.append(line)


def run_team(num_threads: int, log_lines: list = None):
    if log_lines is None:
        log_lines = []
    header = f"--- Forking a team of {num_threads} threads ---"
    print(header)
    log_lines.append(header)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, tid, num_threads, log_lines)
                   for tid in range(num_threads)]
        for f in futures:
            f.result()  # implicit barrier: join

    footer = "--- Joined thread team. Execution returned to serial master ---\n"
    print(footer)
    log_lines.append(footer)
    return log_lines


def task_1_1_demo():
    run_team(4)


def task_1_1_repeat(runs: int = 10, outfile: str = "task1_1_output.txt"):
    all_lines = []
    for run_idx in range(1, runs + 1):
        all_lines.append(f"=== RUN {run_idx} ===")
        run_team(4, all_lines)
    with open(outfile, "w") as f:
        f.write("\n".join(all_lines))
    print(f"Saved {runs} runs to {outfile}")


def task_1_2_sweep(outfile: str = "task1_2_oversubscription.csv"):
    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    results = []
    print(f"{'P':>4} | {'Time (s)':>10}")
    print("-" * 20)
    for p in thread_counts:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p) as executor:
            futures = [executor.submit(lambda: None) for _ in range(p)]
            for f in futures:
                f.result()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        results.append((p, elapsed))
        print(f"{p:>4} | {elapsed:>10.6f}")

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_threads_P", "time_seconds"])
        writer.writerows(results)
    print(f"Saved sweep data to {outfile}")


def cpu_heavy_work(n_sqrt: int = 10_000_000):
    total = 0.0
    for i in range(n_sqrt):
        total += math.sqrt(i + 1.0)
    return total


def task_1_3_saturation(num_threads: int = 4, n_sqrt: int = 10_000_000):
    print(f"Starting CPU saturation test: {num_threads} threads, "
          f"{n_sqrt:,} sqrt ops each.")
    print("Open htop / Task Manager / Activity Monitor NOW to observe core usage.")
    time.sleep(2)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(cpu_heavy_work, n_sqrt) for _ in range(num_threads)]
        for f in futures:
            f.result()
    t1 = time.perf_counter()
    print(f"Done. Elapsed: {t1 - t0:.4f}s")
    print("NOTE: Because of Python's GIL, pure-Python CPU-bound threads will NOT")
    print("achieve true multi-core parallelism (unlike OpenMP/C). Record this")
    print("observation for your report - it's directly relevant to Question 1.4.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "demo":
        task_1_1_demo()
    elif mode == "repeat":
        task_1_1_repeat()
    elif mode == "sweep":
        task_1_2_sweep()
    elif mode == "saturation":
        task_1_3_saturation()
    else:
        print(__doc__)