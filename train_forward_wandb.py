import wandb
import argparse
import os
from utils import get_device
import torch
import blocks
from tqdm import tqdm

from dataset import SymbolForwardDataset
from utils import get_device

def save_model(path, prefix):
    torch.save(proj_in.cpu().state_dict(), os.path.join(path, prefix+"proj_in.ckpt"))
    torch.save(attention.cpu().state_dict(), os.path.join(path, prefix+"attention.ckpt"))
    torch.save(proj_out.cpu().state_dict(), os.path.join(path, prefix+"proj_out.ckpt"))
    wandb.save(os.path.join(path, prefix + "*.ckpt"))
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train attention model using E Z values.")
    parser.add_argument("-id", help="experiment id")
    parser.add_argument("-db", help="dataset path under ./data")
    parser.add_argument("-e", help="epoch", type=int, default=1000)
    parser.add_argument("-bs", help="batch size", type=int, default=128)
    parser.add_argument("-lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("-hs", help="hidden size", type=int, default=128)
    args = parser.parse_args()

    run = wandb.init(entity="colorslab",project="active_exploration", resume="must", id=args.id)
    run = wandb.Api().run("colorslab/active_exploration/" + args.id)

    print(run.config)

    device = get_device()
    data = {}
    data["Z"] = wandb.restore("save/deneme/Z.pt")
    data["E"] = wandb.restore("save/deneme/E.pt")
    data["data"] = wandb.restore("save/deneme/E.pt")

    train_set = SymbolForwardDataset("./data/" + args.db , "", wandb= data)
    val_set = SymbolForwardDataset("./data/" + args.db , "",wandb= data)

    train_loader = torch.utils.data.DataLoader(train_set, args.bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, args.bs, shuffle=True)

    proj_in = blocks.MLP([train_set.precond.size()[-1], args.hs, args.hs])
    attention = torch.nn.MultiheadAttention(embed_dim=args.hs, num_heads=8, batch_first=True)
    proj_out = blocks.MLP([args.hs, args.hs, train_set.effect.size()[-1]])

    proj_in.to(device)
    attention.to(device)
    proj_out.to(device)
    optimizer = torch.optim.Adam(lr=args.lr, params=[
        {"params": proj_in.parameters()},
        {"params": attention.parameters()},
        {"params": proj_out.parameters()}])
    criterion = torch.nn.MSELoss()
    best_loss = 1e100
    os.mkdir(wandb.run.dir+ "/save/attention")
    for e in range(args.e):
        epoch_loss = 0.0
        val_loss = 0.0
        for i, (pre_i, eff_i, m_i) in enumerate(tqdm(train_loader)):
            pre_i = pre_i.to(device)
            eff_i = eff_i.to(device)
            m_i = m_i.to(device)
            
            p_i = proj_in(pre_i)
            h_i, _ = attention(p_i, p_i, p_i, key_padding_mask=~m_i.bool())
            e_bar = proj_out(h_i)
            loss = criterion(e_bar, eff_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= (i+1)

        for i, (pre_i, eff_i, m_i) in enumerate(tqdm(val_loader)):
            pre_i = pre_i.to(device)
            eff_i = eff_i.to(device)
            m_i = m_i.to(device)

            with torch.no_grad():
                p_i = proj_in(pre_i)
                h_i, _ = attention(p_i, p_i, p_i, key_padding_mask=~m_i.bool())
                e_bar = proj_out(h_i)
            val_loss += criterion(e_bar, eff_i)
        val_loss /= (i+1)
        if val_loss < best_loss:
            save_model(wandb.run.dir + "/save/attention", "best_")
        save_model(wandb.run.dir+ "/save/attention", "last_")
        proj_in.to(device)
        attention.to(device)
        proj_out.to(device)

        print(f"Epoch={e+1}, train loss={epoch_loss}, val loss={val_loss}")