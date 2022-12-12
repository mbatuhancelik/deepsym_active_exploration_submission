"""Train DeepSym"""
import argparse

import torch

import utils
from dataset import StateActionEffectDataset

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-c", "--config", help="config file", type=str, required=True)
args = parser.parse_args()

config = utils.parse_and_init(args)
model = utils.create_model_from_config(config)
model.print_model()
for name in model.module_names:
    print(f"{name} params={utils.get_parameter_count(getattr(model, name)):,}")


train_set = StateActionEffectDataset(config["data"], split="train")
val_set = StateActionEffectDataset(config["data"], split="val")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, num_workers=4, shuffle=True,
                                           collate_fn=utils.collate_fn)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, num_workers=4, shuffle=True,
                                         collate_fn=utils.collate_fn)
model.train(args.e, train_loader, val_loader)
