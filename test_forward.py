import os
import argparse

import torch
import blocks
from tqdm import tqdm

from dataset import SymbolForwardDataset
from utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train a symbol forward model")
    parser.add_argument("-s", help="path", type=str, required=True)
    parser.add_argument("-hs", help="hidden size", type=int, default=128)
    args = parser.parse_args()

    device = get_device()

    test_set = SymbolForwardDataset(args.s, "test_")
    test_loader = torch.utils.data.DataLoader(test_set, args.bs, shuffle=True)

    proj_in = blocks.MLP([20, args.hs, args.hs])
    attention = torch.nn.MultiheadAttention(embed_dim=args.hs, num_heads=8, batch_first=True)
    proj_out = blocks.MLP([args.hs, args.hs, 8])

    proj_in.load_state_dict(torch.load(os.path.join(args.s, "best_proj_in.ckpt")))
    attention.load_state_dict(torch.load(os.path.join(args.s, "best_attention.ckpt")))
    proj_out.load_state_dict(torch.load(os.path.join(args.s, "best_proj_out.ckpt")))

    proj_in.to(device)
    attention.to(device)
    proj_out.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss = 0.0
    accuracy = 0
    for i, (pre_i, eff_i, m_i) in enumerate(tqdm(test_loader)):
        pre_i = pre_i.to(device)
        eff_i = eff_i.to(device)
        m_i = m_i.to(device)

        with torch.no_grad():
            p_i = proj_in(pre_i)
            h_i, _ = attention(p_i, p_i, p_i, key_padding_mask=~m_i.bool())
            e_bar = proj_out(h_i)
        test_loss += criterion(e_bar, eff_i)
        acc = ((torch.sigmoid(e_bar).round()-eff_i).abs().sum(dim=-1) < 0.1).sum()
        accuracy += acc
    test_loss /= (i+1)
    print(f"Test loss={test_loss} accuracy={accuracy/len(test_set)}")
