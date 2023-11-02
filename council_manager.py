import os
import torch

PATH="./council"

def load(path=PATH):
    council_list = os.listdir("./council")
    assert(len(council_list)> 2)
    council = []
    for m in council_list:
        council.append(torch.load(f"./council/{m}"))
        council[-1].to("cpu")
    return council
def save(council ,path=PATH):
    for i, m in enumerate(council):
        torch.save(m, f"{PATH}/{i}.pt")


        