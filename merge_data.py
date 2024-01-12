import os
import argparse

import torch
from dataset_utils import metrics_by_name

parser = argparse.ArgumentParser("Merge exploration data.")
parser.add_argument("-o", help="dataset name", type=str, required=True)
parser.add_argument("-i", help="number of rolls", type=int, required=True)
args = parser.parse_args()

keys = ["action", "effect", "mask", "state", "post_state"]

output_folder = os.path.join("./data", args.o)
for key in keys:
    tensors = []
    for i in range(args.i):
        try:
            tensors.append(torch.load(os.path.join(output_folder, f"{key}_{i}.pt")))
        except:
            continue
        finally:
            pass
    field = torch.cat(tensors, dim=0)
    torch.save(field, os.path.join(os.path.join(output_folder, f"{key}.pt")))

metrics_by_name(args.o)
