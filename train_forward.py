import os
import argparse

import torch
import wandb

from utils import get_device, get_parameter_count
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


train_set = load_dataset("train", run, device)
val_set = load_dataset("val", run, device)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.b, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.b, shuffle=False)

criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())

for e in range(args.e):
    train_obj_loss = 0.0
    train_rel_loss = 0.0
    for zo_i, zr_i, a, zo_f, zr_f, m in train_loader:
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

        train_obj_loss += obj_loss.item()
        train_rel_loss += rel_loss.item()

    train_obj_loss = train_obj_loss / len(train_loader)
    train_rel_loss = train_rel_loss / len(train_loader)

    val_obj_loss = 0.0
    val_rel_loss = 0.0
    with torch.no_grad():
        for zo_i, zr_i, a, zo_f, zr_f, m in val_loader:
            zi_cat = torch.cat([zo_i, a], dim=-1)
            zo_f_bar, zo_r_bar = model(zi_cat, zr_i)
            m = m.unsqueeze(2)
            m_rel = (m @ m.permute(0, 2, 1)).unsqueeze(1)
            obj_loss = (criterion(zo_f_bar, zo_f) * m).sum(dim=[1, 2]).mean()
            rel_loss = (criterion(zo_r_bar, zr_f) * m_rel).sum(dim=[1, 2, 3]).mean()

            val_obj_loss += obj_loss.item()
            val_rel_loss += rel_loss.item()

    val_obj_loss = val_obj_loss / len(val_loader)
    val_rel_loss = val_rel_loss / len(val_loader)

    wandb.log({"train_obj_loss": train_obj_loss,
               "train_rel_loss": train_rel_loss,
               "val_obj_loss": val_obj_loss,
               "val_rel_loss": val_rel_loss})

    print(f"Epoch={e}, Train obj loss={train_obj_loss:.5f}, Train rel loss={train_rel_loss:.5f}, "
          f"Val obj loss={val_obj_loss:.5f}, Val rel loss={val_rel_loss:.5f}")

    sd = model.eval().cpu().state_dict()
    torch.save(sd, save_path)
    model.train().to(device)
wandb.save(save_path)
