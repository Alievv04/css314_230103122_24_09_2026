import sys
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

WIDTH, HEIGHT = 960, 540
MAX_ITER = 1000


if HAVE_NUMBA:
    import numpy as np

    @njit
    def compute_pixel(px, py, width, height, max_iter):
        x0 = (px - width / 2.0) * 4.0 / width
        y0 = (py - height / 2.0) * 4.0 / height
        x, y = 0.0, 0.0
        iteration = 0
        while x * x + y * y <= 4.0 and iteration < max_iter:
            xtemp = x * x - y * y + x0
            y = 2.0 * x * y + y0
            x = xtemp
            iteration += 1
        return iteration

    @njit(parallel=True)
    def render_mandelbrot_static(width, height, max_iter):
        img = np.zeros((height, width), dtype=np.int32)
        for y in prange(height):
            for x in range(width):
                img[y, x] = compute_pixel(x, y, width, height, max_iter)
        return img
else:
    def compute_pixel(px, py, width, height, max_iter):
        x0 = (px - width / 2.0) * 4.0 / width
        y0 = (py - height / 2.0) * 4.0 / height
        x, y = 0.0, 0.0
        iteration = 0
        while x * x + y * y <= 4.0 and iteration < max_iter:
            xtemp = x * x - y * y + x0
            y = 2.0 * x * y + y0
            x = xtemp
            iteration += 1
        return iteration


def render_row(y, width, height, max_iter):
    return [compute_pixel(x, y, width, height, max_iter) for x in range(width)]


def render_static(width, height, max_iter, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        rows = list(executor.map(
            lambda y: render_row(y, width, height, max_iter),
            range(height)
        ))
    return rows


def render_dynamic(width, height, max_iter, num_threads, chunk_size):
    row_counter = [0]
    lock = Lock()
    work_counts = [0] * num_threads

    def worker(thread_idx):
        local_rows_done = 0
        while True:
            with lock:
                start = row_counter[0]
                if start >= height:
                    break
                end = min(start + chunk_size, height)
                row_counter[0] = end
            for y in range(start, end):
                render_row(y, width, height, max_iter)
            local_rows_done += (end - start)
        work_counts[thread_idx] = local_rows_done

    threads_list = []
    import threading
    for t in range(num_threads):
        th = threading.Thread(target=worker, args=(t,))
        threads_list.append(th)
        th.start()
    for th in threads_list:
        th.join()

    return work_counts


def task_3_1_construction():
    print("Static scheduling:")
    t0 = time.perf_counter()
    render_static(WIDTH, HEIGHT, MAX_ITER, 4)
    t1 = time.perf_counter()
    print(f"Time = {t1 - t0:.4f}s")

    print("Dynamic scheduling:")
    t2 = time.perf_counter()
    render_dynamic(WIDTH, HEIGHT, MAX_ITER, 4, 16)
    t3 = time.perf_counter()
    print(f"Time = {t3 - t2:.4f}s")


def task_3_2_sweep(outfile="task3_2_sweep.csv"):
    thread_counts = [2, 4, 8, 16]
    chunk_sizes = [1, 16, 64, 256]
    results = []

    print(f"{'P':>4} | {'Chunk':>6} | {'AvgTime':>10}")
    for p in thread_counts:
        for c in chunk_sizes:
            times = []
            for trial in range(3):
                t0 = time.perf_counter()
                render_dynamic(WIDTH, HEIGHT, MAX_ITER, p, c)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            avg_time = sum(times) / len(times)
            results.append((p, c, avg_time))
            print(f"{p:>4} | {c:>6} | {avg_time:>10.4f}")

    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads_P", "chunk_size", "avg_time_seconds"])
        w.writerows(results)
    print(f"Saved to {outfile}")


def task_3_4_imbalance(num_threads=4, chunk_size=16):
    work_counts = render_dynamic(WIDTH, HEIGHT, MAX_ITER, num_threads, chunk_size)
    max_work = max(work_counts)
    min_work = min(work_counts)
    avg_work = sum(work_counts) / len(work_counts)
    imbalance = (max_work - min_work) / avg_work if avg_work > 0 else 0

    print(f"Rows per thread: {work_counts}")
    print(f"Max={max_work}, Min={min_work}, Avg={avg_work:.2f}")
    print(f"Load Imbalance = {imbalance:.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "construction"
    if not HAVE_NUMBA:
        print("numba not installed, using pure Python compute_pixel.")

    if mode == "construction":
        task_3_1_construction()
    elif mode == "sweep":
        task_3_2_sweep()
    elif mode == "imbalance":
        task_3_4_imbalance()