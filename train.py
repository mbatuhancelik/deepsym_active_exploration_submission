"""Train DeepSym"""
import argparse

import torch

import utils
from dataset import StateActionEffectDataset

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-c", "--config", help="config file", type=str, required=True)
parser.add_argument("-cid", help="council_id", type=str, required=False)
args = parser.parse_args()

config = utils.parse_and_init(args)
model = utils.create_model_from_config(config)
model.print_model()
for name in model.module_names:
    print(f"{name} params={utils.get_parameter_count(getattr(model, name)):,}")


train_set = StateActionEffectDataset(config["dataset_name"], split="train")
val_set = StateActionEffectDataset(config["dataset_name"], split="val")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"])
model.train(config["epoch"], train_loader, val_loader)
if args.cid:
    model.load("_best",from_wandb=True)
    model.to("cpu")
    torch.save(model, f"./council/{args.cid}.pt")
utils.wandb_finalize()
