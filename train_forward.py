import os
import argparse

import torch
import wandb

from utils import get_device, get_parameter_count
import models


parser = argparse.ArgumentParser("Train symbol forward model")
parser.add_argument("-i", help="Wandb run id", type=str)
parser.add_argument("-n", help="Number of hidden units", type=int)
parser.add_argument("-l", help="Number of layers", type=int)
parser.add_argument("-e", help="Number of epochs", type=int)
parser.add_argument("-b", help="Batch size", type=int)
parser.add_argument("-lr", help="Learning rate", type=float)
args = parser.parse_args()

run = wandb.init(entity="colorslab", project="multideepsym", resume="must", id=args.i)
device = get_device()
wandb.config.update({"device": device}, allow_val_change=True)

# load the data from wandb
z_obj_pre = torch.load(wandb.restore(os.path.join(run.config["save_folder"], "z_obj_pre.pt")).name).to(device)
z_rel_pre = torch.load(wandb.restore(os.path.join(run.config["save_folder"], "z_rel_pre.pt")).name).to(device)
z_act = torch.load(wandb.restore(os.path.join(run.config["save_folder"], "z_act.pt")).name).to(device)
z_obj_post = torch.load(wandb.restore(os.path.join(run.config["save_folder"], "z_obj_post.pt")).name).to(device)
z_rel_post = torch.load(wandb.restore(os.path.join(run.config["save_folder"], "z_rel_post.pt")).name).to(device)
mask = torch.load(wandb.restore(os.path.join(run.config["save_folder"], "mask.pt")).name).to(device)

input_dim = run.config["latent_dim"] + run.config["action_dim"]
model = models.SymbolForward(input_dim=input_dim, hidden_dim=args.n,
                             output_dim=run.config["latent_dim"], num_layers=args.l,
                             num_heads=run.config["n_attention_heads"]).to(device)
print(model)
print(f"Number of parameters: {get_parameter_count(model)}")
wandb.config.update({"forward_model":
                     {"hidden_unit": args.n,
                      "layer": args.l,
                      "epoch": args.e,
                      "batch_size": args.b,
                      "learning_rate": args.lr}}, allow_val_change=True)
save_path = os.path.join(run.config["save_folder"], "symbol_forward.pt")
if not os.path.exists(run.config["save_folder"]):
    os.makedirs(run.config["save_folder"])

dataset = torch.utils.data.TensorDataset(z_obj_pre, z_rel_pre, z_act,
                                         z_obj_post, z_rel_post, mask)

loader = torch.utils.data.DataLoader(dataset, batch_size=args.b, shuffle=True)
criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())

for e in range(args.e):
    avg_obj_loss = 0.0
    avg_rel_loss = 0.0
    for zo_i, zr_i, a, zo_f, zr_f, m in loader:
        zi_cat = torch.cat([zo_i, a], dim=-1)
        zo_f_bar, zo_r_bar = model(zi_cat, zr_i)
        m = m.unsqueeze(2)
        m_rel = (m @ m.permute(0, 2, 1)).unsqueeze(1)
        obj_loss = (criterion(zo_f_bar, zo_f) * m).sum(dim=[1, 2]).mean()
        rel_loss = (criterion(zo_r_bar, zr_f) * m_rel).sum(dim=[1, 2, 3]).mean()
        loss = obj_loss + rel_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_obj_loss += obj_loss.item()
        avg_rel_loss += rel_loss.item()
    print(f"Epoch={e}, Obj loss={avg_obj_loss/len(loader):.5f}, Rel loss={avg_rel_loss/len(loader):.5f}")

    sd = model.eval().cpu().state_dict()
    torch.save(sd, save_path)
    model.train().to(device)
wandb.save(save_path)
