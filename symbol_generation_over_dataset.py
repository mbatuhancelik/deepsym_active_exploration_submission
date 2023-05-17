import os
import sys

import wandb
import torch
import utils
from dataset import StateActionEffectDataset


run_id = sys.argv[1]
run = wandb.init(entity="colorslab", project="multideepsym", resume="must", id=run_id)
model = utils.create_model_from_config(run.config)
model.load("_best", from_wandb=True)
model.print_model()

for name in model.module_names:
    getattr(model, name).eval()


dataset = StateActionEffectDataset(run.config["dataset_name"], split="val")
loader = torch.utils.data.DataLoader(dataset, batch_size=run.config["batch_size"])

z_obj_pre = []
z_rel_pre = []
z_act = []
z_obj_post = []
z_rel_post = []
mask = []
for sample in loader:
    with torch.no_grad():
        z_i = model.encode(sample["state"], eval_mode=True)
        zr_i = model.attn_weights(sample["state"], sample["pad_mask"], eval_mode=True)
        z_f = model.encode(sample["post_state"], eval_mode=True)
        zr_f = model.attn_weights(sample["post_state"], sample["pad_mask"], eval_mode=True)
        z_obj_pre.append(z_i.cpu())
        z_rel_pre.append(zr_i.cpu())
        z_act.append(sample["action"].cpu())
        z_obj_post.append(z_f.cpu())
        z_rel_post.append(zr_f.cpu())
        mask.append(sample["pad_mask"].cpu())

z_obj_pre = torch.cat(z_obj_pre, axis=0)
z_rel_pre = torch.cat(z_rel_pre, axis=0)
z_act = torch.cat(z_act, axis=0)
z_obj_post = torch.cat(z_obj_post, axis=0)
z_rel_post = torch.cat(z_rel_post, axis=0)
mask = torch.cat(mask, axis=0)

if not os.path.exists(run.config["save_folder"]):
    os.makedirs(run.config["save_folder"])

torch.save(z_obj_pre, os.path.join(run.config["save_folder"], "z_obj_pre.pt"))
torch.save(z_rel_pre, os.path.join(run.config["save_folder"], "z_rel_pre.pt"))
torch.save(z_act, os.path.join(run.config["save_folder"], "z_act.pt"))
torch.save(z_obj_post, os.path.join(run.config["save_folder"], "z_obj_post.pt"))
torch.save(z_rel_post, os.path.join(run.config["save_folder"], "z_rel_post.pt"))
torch.save(mask, os.path.join(run.config["save_folder"], "mask.pt"))

wandb.save(os.path.join(run.config["save_folder"], "z_obj_pre.pt"))
wandb.save(os.path.join(run.config["save_folder"], "z_rel_pre.pt"))
wandb.save(os.path.join(run.config["save_folder"], "z_act.pt"))
wandb.save(os.path.join(run.config["save_folder"], "z_obj_post.pt"))
wandb.save(os.path.join(run.config["save_folder"], "z_rel_post.pt"))
wandb.save(os.path.join(run.config["save_folder"], "mask.pt"))

run.finish()
