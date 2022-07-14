import os
import argparse

import torch

from models import DeepSymv3
from dataset import SegmentedSAEFolder
import blocks
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser("record symbols.")
    parser.add_argument("-d", help="data path", type=str, required=True)
    parser.add_argument("-s", help="model path", type=str, required=True)
    args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

STATE_BITS = 8
ACTION_BITS = 12
BN = True

encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.MLP([512, STATE_BITS]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

projector = torch.nn.Linear(STATE_BITS+ACTION_BITS, 128)
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
decoder_att = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
decoder = blocks.MLP([128, 256, 256, 256, 3], batch_norm=BN)

encoder.to(device)
decoder.to(device)
projector.to(device)
decoder_att.to(device)

model = DeepSymv3(encoder=encoder, decoder=decoder, decoder_att=decoder_att, projector=projector,
                  device=device, lr=0.0001, path=args.s, coeff=1.0)
model.load("_best")
model.eval_mode()

model.print_model()
for name in model.module_names:
    print(f"{name} params={utils.get_parameter_count(getattr(model, name)):,}")
    for param in getattr(model, name).parameters():
        param.requires_grad = False

valid_objects = {i: True for i in range(4, 7)}
train_set = SegmentedSAEFolder(args.d, max_pad=3, valid_objects=valid_objects, normalize=True, old=False, partitions=list(range(10)), with_post=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

z_precond = []
z_effect = []
mask = []

for i, sample in enumerate(train_loader):
    z_i = model.concat(sample, eval_mode=True)
    z_f = model.encode(sample["post_state"], sample["post_pad_mask"], eval_mode=True)
    z_precond.append(z_i)
    z_effect.append(z_f)
    mask.append(sample["pad_mask"])

z_precond = torch.cat(z_precond, dim=0)
z_effect = torch.cat(z_effect, dim=0)
mask = torch.cat(mask, dim=0)
torch.save(z_precond, os.path.join(args.s, "z_precond.pt"))
torch.save(z_effect, os.path.join(args.s, "z_effect.pt"))
torch.save(mask, os.path.join(args.s, "mask.pt"))
