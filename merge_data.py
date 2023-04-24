import os
import argparse

import torch
from dataset_utils import metrics_by_name

parser = argparse.ArgumentParser("Merge exploration data.")
parser.add_argument("-o", help="dataset name", type=str, required=True)
parser.add_argument("-i", help="number of rolls", type=int, required=True)
args = parser.parse_args()

keys = ["action", "effect", "mask", "state"]

output_folder = os.path.join("./data", args.o)
for key in keys:
    field = torch.cat([torch.load(os.path.join(output_folder, f"{key}_{i}.pt")) for i in range(args.i)], dim=0)
    torch.save(field, os.path.join(os.path.join(output_folder, f"{key}.pt")))
    for i in range(args.i):
        os.remove(os.path.join(output_folder, f"{key}_{i}.pt"))

metrics_by_name(args.o)



