import time
import os
import argparse
import subprocess
from multiprocessing import Process


def collect(num, folder, idx):
    subprocess.run(["python", "explore.py", "-N", num, "-o", folder, "-i", idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train DeepSym.")
    parser.add_argument("-d", help="data folder", type=str, required=True)
    parser.add_argument("-N", help="number of data per proc", type=int, required=True)
    parser.add_argument("-p", help="number of procs", type=int, required=True)
    args = parser.parse_args()
    procs = []
    start = time.time()
    for i in range(args.p):
        p = Process(target=collect, args=[str(args.N), args.d, str(i)])
        p.start()
        procs.append(p)

    for i in range(args.p):
        procs[i].join()
    end = time.time()
    elapsed = end - start
    print(args.N, file=open(os.path.join(args.d, "info.txt"), "w"))
    print(args.p, file=open(os.path.join(args.d, "info.txt"), "a"))
    print(f"Collected {args.p*args.N} samples in {elapsed:.2f} seconds. {args.p*args.N/elapsed}")
