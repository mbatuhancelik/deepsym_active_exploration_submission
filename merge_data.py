import os
import argparse

import torch


parser = argparse.ArgumentParser("Merge exploration data.")
parser.add_argument("-o", help="output folder", type=str, required=True)
parser.add_argument("-i", help="number of rolls", type=int, required=True)
args = parser.parse_args()

keys = ["action", "effect", "mask", "state"]

for key in keys:
    field = torch.cat([torch.load(os.path.join(args.o, f"{key}_{i}.pt")) for i in range(args.i)], dim=0)
    torch.save(field, os.path.join(os.path.join(args.o, f"{key}.pt")))
    for i in range(args.i):
        os.remove(os.path.join(args.o, f"{key}_{i}.pt"))
