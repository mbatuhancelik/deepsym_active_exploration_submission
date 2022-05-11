import time
import argparse
import subprocess
from multiprocessing import Process


def collect(num, folder, idx):
    subprocess.run(["python", "explore.py", "-N", num, "-o", folder, "-i", idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train DeepSym.")
    parser.add_argument("-d", help="data folder", type=str, required=True)
    parser.add_argument("-N", help="number of data per proc", type=int, required=True)
    n_proc = 8
    args = parser.parse_args()
    procs = []
    start = time.time()
    for i in range(n_proc):
        p = Process(target=collect, args=[str(args.N), args.d, str(i*args.N)])
        p.start()
        procs.append(p)

    for i in range(n_proc):
        procs[i].join()
    end = time.time()
    elapsed = end - start
    print(f"Collected {n_proc*args.N} samples in {elapsed:.2f} seconds. {n_proc*args.N/elapsed}")
