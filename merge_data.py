import time
import os
import argparse

import torch
import numpy as np


parser = argparse.ArgumentParser("Merge exploration data.")
parser.add_argument("-o", help="output folder", type=str, required=True)
parser.add_argument("-i", help="number of rolls", type=int, required=True)
args = parser.parse_args()

keys = ["action", "effect", "mask", "state"]

data = {}
for key in keys:
    data[key] = torch.load(f"{args.o}/{key}_0.pt")


for i in range(1,args.i):
    for key in keys:
        dk = torch.load(f"{args.o}/{key}_{i}.pt")
        data[key] = torch.cat([data[key], dk], axis = 0)

for key in keys:
    torch.save(data[key], f"{args.o}/{key}.pt")

for i in range(args.i):
    for key in keys:
        os.remove(f"{args.o}/{key}_{i}.pt")
        