import argparse

import torch
import blocks
from tqdm import tqdm

from dataset import SymbolForwardDataset
from utils import get_device


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train a symbol forward model")
    parser.add_argument("-s", help="path", type=str, required=True)
    parser.add_argument("-e", help="epoch", type=int, default=1000)
    parser.add_argument("-bs", help="batch size", type=int, default=128)
    parser.add_argument("-lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("-hs", help="hidden size", type=int, default=128)
    args = parser.parse_args()

    device = get_device()

    train_set = SymbolForwardDataset(args.s, "train_")
    val_set = SymbolForwardDataset(args.s, "val_")
    train_loader = torch.utils.data.DataLoader(train_set, args.bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, args.bs, shuffle=True)

    proj_in = blocks.MLP([20, args.hs, args.hs])
    attention = torch.nn.MultiheadAttention(embed_dim=args.hs, num_heads=8, batch_first=True)
    proj_out = blocks.MLP([args.hs, args.hs, 8])

    proj_in.to(device)
    attention.to(device)
    proj_out.to(device)

    optimizer = torch.optim.Adam(lr=args.lr, params=[
            {"params": proj_in.parameters()},
            {"params": attention.parameters()},
            {"params": proj_out.parameters()}])
    criterion = torch.nn.BCEWithLogitsLoss()

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
        print(f"Epoch={e+1}, train loss={epoch_loss}, val loss={val_loss}")

