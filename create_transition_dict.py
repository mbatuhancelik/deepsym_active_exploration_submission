import pickle
import os
import argparse

import wandb
import torch

import utils
from dataset import load_symbol_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Wandb run id", type=str)
args = parser.parse_args()

run = wandb.init(entity="colorslab", project="multideepsym", resume="must", id=args.i)
device = utils.get_device()
wandb.config.update({"device": device}, allow_val_change=True)

train_set = load_symbol_dataset("train", run, device)
val_set = load_symbol_dataset("val", run, device)
loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)

dictionary = {}

for i, (obj_pre, rel_pre, action, obj_post, rel_post, mask) in enumerate(loader):
    valid_objects = torch.where(mask > 0)[1].tolist()
    permuted_ary = utils.permute(valid_objects)
    for permutation in permuted_ary:
        permuted = utils.permute_symbols(obj_pre[0], rel_pre[0], action[0],
                                         obj_post[0], rel_post[0], permutation)
        obj_pre_p, rel_pre_p, action_p, obj_post_p, rel_post_p = permuted
        key = (obj_pre_p, rel_pre_p)
        value = (obj_post_p, rel_post_p)
        if key in dictionary:
            if action_p in dictionary[key]:
                if value in dictionary[key][action_p]:
                    dictionary[key][action_p][value] += 1
                else:
                    dictionary[key][action_p][value] = 1
            else:
                dictionary[key][action_p] = {value: 1}
        else:
            dictionary[key] = {action_p: {value: 1}}
    if i % 100 == 0:
        print(f"Processed {i} / {len(loader)}")

if not os.path.exists(run.config["save_folder"]):
    os.makedirs(run.config["save_folder"])
save_path = os.path.join(run.config["save_folder"], "transition.pkl")
target_file = open(save_path, "wb")
pickle.dump(dictionary, target_file)
target_file.close()
wandb.save(os.path.join(run.config["save_folder"], "transition.pkl"))
