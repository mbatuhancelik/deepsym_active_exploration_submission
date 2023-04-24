import torch
import numpy as np
from dataset import StateActionEffectDataset
import argparse


def metrics_by_name(dataset_name):
    train = StateActionEffectDataset(dataset_name, "train")
    val = StateActionEffectDataset(dataset_name, "val")
    total = StateActionEffectDataset(dataset_name, "")
    print("====TRAIN SET=====")
    metrics(train)
    print("====VAL SET=====")
    metrics(val)
    print("====TOTAL=====")
    metrics(total)
def metrics(dataset):
    print(f"{len(dataset)} samples \n")
    count = 0
    for e in dataset.effect:
        if torch.any(e[:,2].abs().max() > 0.1):
            count += 1
    print(f"{count / len(dataset) * 100}% resulted in ds")
    print(f"{count } samples")
    count = 0
    for e in dataset.effect:
        if np.count_nonzero(e[:,2] > 0.1) == 2:
            count += 1
    print(f"{count / len(dataset) * 100:.2f}% multi object(2) movement")
    print(f"{count } samples")
    count = 0
    for e in dataset.effect:
        if np.count_nonzero(e[:,2] > 0.1) > 2:
            count += 1
    print(f"{count / len(dataset) * 100:.2f}% multi object(3) movement")
    print(f"{count } samples")
    count = 0
    for e in dataset.effect:
        if np.count_nonzero(e[:,2] < -0.1) > 0 or np.count_nonzero(e[:,2 + 9] < -0.1) > 1:
            count += 1
    print(f"{count / len(dataset) * 100:.2f}% towers falling down")
    print(f"{count } samples")
    count = 0
    for sample in dataset:
        target = torch.argmin(sample["action"][0])
        if sample["effect"][target][2] < 0.1:
            if torch.any((((sample["action"][:,0]) != -1).view((-1,1)) * sample["effect"])[:,2] > 0.2):
                count += 1
    print(f"{count / len(dataset) * 100:.2f}% mistargets")
    print(f"{count } samples")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("See dataset metrics.")
    parser.add_argument("-o", help="dataset name", type=str, required=True)
    args = parser.parse_args()
    metrics_by_name(args.o)