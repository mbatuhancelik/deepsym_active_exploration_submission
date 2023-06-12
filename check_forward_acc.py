import os

import torch
import wandb

import models


def load_dataset(name, run, device):
    z_obj_pre = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_obj_pre.pt")).name).to(device)
    z_rel_pre = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_rel_pre.pt")).name).to(device)
    z_act = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_act.pt")).name).to(device)
    z_obj_post = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_obj_post.pt")).name).to(device)
    z_rel_post = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_rel_post.pt")).name).to(device)
    mask = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_mask.pt")).name).to(device)

    dataset = torch.utils.data.TensorDataset(z_obj_pre, z_rel_pre, z_act,
                                             z_obj_post, z_rel_post, mask)
    return dataset


run_id = "mhd6ocdg"
run = wandb.init(entity="colorslab", project="multideepsym", resume="must", id=run_id)
wandb.config.update({"device": "cpu"}, allow_val_change=True)
config = dict(run.config)


sym_model = models.SymbolForward(input_dim=run.config["latent_dim"]+run.config["action_dim"],
                                 hidden_dim=run.config["forward_model"]["hidden_unit"],
                                 output_dim=run.config["latent_dim"],
                                 num_layers=run.config["forward_model"]["layer"],
                                 num_heads=run.config["n_attention_heads"])
path = os.path.join(run.config["save_folder"], "symbol_forward.pt")
module_dict = torch.load(wandb.restore(path).name)
sym_model.load_state_dict(module_dict)

for p in sym_model.parameters():
    p.requires_grad = False

train_set = load_dataset("train", run, "cpu")
val_set = load_dataset("val", run, "cpu")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)

obj_single = 0
obj_full = 0
rel_single = 0
rel_full = 0
N = 0
Nb = 0
for zo_i, zr_i, a, zo_f, zr_f, m in train_loader:
    zi_cat = torch.cat([zo_i, a], dim=-1)
    zo_f_bar, zo_r_bar = sym_model(zi_cat, zr_i)
    m = m.unsqueeze(2)
    m_rel = (m @ m.permute(0, 2, 1)).unsqueeze(1)
    obj_acc = torch.abs(zo_f - zo_f_bar.sigmoid().round()) < 1e-3
    rel_acc = torch.abs(zr_f - zo_r_bar.sigmoid().round()) < 1e-3
    obj_single += torch.all(obj_acc, dim=-1).sum()
    obj_full += torch.all(torch.all(obj_acc, dim=-1), dim=-1).sum()
    rel_single += torch.all(rel_acc, dim=-1).sum()
    rel_full += torch.all(torch.all(torch.all(rel_acc, dim=-1), dim=-1), dim=-1).sum()
    N += m.sum()
    Nb += m.shape[0]

print(f"Train Single object accuracy={obj_single/N}")
print(f"Train Full object accuracy={obj_full/Nb}")
print(f"Train Single relation accuracy={rel_single/(N*run.config['n_attention_heads'])}")
print(f"Train Full relation accuracy={rel_full/Nb}")

obj_single = 0
obj_full = 0
rel_single = 0
rel_full = 0
N = 0
Nb = 0
for zo_i, zr_i, a, zo_f, zr_f, m in val_loader:
    zi_cat = torch.cat([zo_i, a], dim=-1)
    zo_f_bar, zo_r_bar = sym_model(zi_cat, zr_i)
    m = m.unsqueeze(2)
    m_rel = (m @ m.permute(0, 2, 1)).unsqueeze(1)
    obj_acc = torch.abs(zo_f - zo_f_bar.sigmoid().round()) < 1e-3
    rel_acc = torch.abs(zr_f - zo_r_bar.sigmoid().round()) < 1e-3
    obj_single += torch.all(obj_acc, dim=-1).sum()
    obj_full += torch.all(torch.all(obj_acc, dim=-1), dim=-1).sum()
    rel_single += torch.all(rel_acc, dim=-1).sum()
    rel_full += torch.all(torch.all(torch.all(rel_acc, dim=-1), dim=-1), dim=-1).sum()
    N += m.sum()
    Nb += 1

print(f"Val Single object accuracy={obj_single/N}")
print(f"Val Full object accuracy={obj_full/N}")
print(f"Val Single relation accuracy={rel_single/(N*run.config['n_attention_heads'])}")
print(f"Val Full relation accuracy={rel_full/N}")
