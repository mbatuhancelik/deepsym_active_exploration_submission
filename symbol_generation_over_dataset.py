import os
import sys

import wandb
import torch
import utils
from dataset import StateActionEffectDataset


def encode_dataset(loader, model):
    z_obj_pre = []
    z_rel_pre = []
    z_act = []
    z_obj_post = []
    z_rel_post = []
    mask = []
    for i, sample in enumerate(loader):
        with torch.no_grad():
            z_i = model.encode(sample["state"], eval_mode=True)
            zr_i = model.attn_weights(sample["state"], sample["pad_mask"], eval_mode=True)
            z_f = model.encode(sample["post_state"], eval_mode=True)
            zr_f = model.attn_weights(sample["post_state"], sample["pad_mask"], eval_mode=True)
            z_obj_pre.append(z_i.cpu().bool())
            z_rel_pre.append(zr_i.cpu().bool())
            z_act.append(sample["action"].cpu().char())
            z_obj_post.append(z_f.cpu().bool())
            z_rel_post.append(zr_f.cpu().bool())
            mask.append(sample["pad_mask"].cpu().bool())
        if i % 100 == 0:
            print(f"Encoded {i} samples out of {len(loader.dataset)}")

    z_obj_pre = torch.cat(z_obj_pre, axis=0)
    z_rel_pre = torch.cat(z_rel_pre, axis=0)
    z_act = torch.cat(z_act, axis=0)
    z_obj_post = torch.cat(z_obj_post, axis=0)
    z_rel_post = torch.cat(z_rel_post, axis=0)
    mask = torch.cat(mask, axis=0)
    return z_obj_pre, z_rel_pre, z_act, z_obj_post, z_rel_post, mask


def save_and_upload(z_obj_pre, z_rel_pre, z_act, z_obj_post, z_rel_post, mask, run, name):
    torch.save(z_obj_pre, os.path.join(run.config["save_folder"], f"{name}_z_obj_pre.pt"))
    torch.save(z_rel_pre, os.path.join(run.config["save_folder"], f"{name}_z_rel_pre.pt"))
    torch.save(z_act, os.path.join(run.config["save_folder"], f"{name}_z_act.pt"))
    torch.save(z_obj_post, os.path.join(run.config["save_folder"], f"{name}_z_obj_post.pt"))
    torch.save(z_rel_post, os.path.join(run.config["save_folder"], f"{name}_z_rel_post.pt"))
    torch.save(mask, os.path.join(run.config["save_folder"], f"{name}_mask.pt"))

    wandb.save(os.path.join(run.config["save_folder"], f"{name}_z_obj_pre.pt"))
    wandb.save(os.path.join(run.config["save_folder"], f"{name}_z_rel_pre.pt"))
    wandb.save(os.path.join(run.config["save_folder"], f"{name}_z_act.pt"))
    wandb.save(os.path.join(run.config["save_folder"], f"{name}_z_obj_post.pt"))
    wandb.save(os.path.join(run.config["save_folder"], f"{name}_z_rel_post.pt"))
    wandb.save(os.path.join(run.config["save_folder"], f"{name}_mask.pt"))


run_id = sys.argv[1]
run = wandb.init(entity="colorslab", project="multideepsym", resume="must", id=run_id)
model = utils.create_model_from_config(run.config)
model.load("_best", from_wandb=True)
model.print_model()

for name in model.module_names:
    getattr(model, name).eval()


train_set = StateActionEffectDataset(run.config["dataset_name"], split="train")
val_set = StateActionEffectDataset(run.config["dataset_name"], split="val")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.config["batch_size"])
val_loader = torch.utils.data.DataLoader(val_set, batch_size=run.config["batch_size"])

if not os.path.exists(run.config["save_folder"]):
    os.makedirs(run.config["save_folder"])

z_obj_pre, z_rel_pre, z_act, z_obj_post, z_rel_post, mask = encode_dataset(train_loader, model)
save_and_upload(z_obj_pre, z_rel_pre, z_act, z_obj_post, z_rel_post, mask, run, "train")

z_obj_pre, z_rel_pre, z_act, z_obj_post, z_rel_post, mask = encode_dataset(val_loader, model)
save_and_upload(z_obj_pre, z_rel_pre, z_act, z_obj_post, z_rel_post, mask, run, "val")

run.finish()
