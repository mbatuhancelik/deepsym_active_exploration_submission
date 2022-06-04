"""Train DeepSym"""
import argparse
import os

import torch

import blocks
from models import DeepSymv3
from dataset import SegmentedSAEFolder
from utils import get_parameter_count

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-s", help="save folder", type=str, required=True)
parser.add_argument("-d", help="data folder", type=str, required=True)
parser.add_argument("-state_bits", help="state bits", type=int, required=True)
parser.add_argument("-action_bits", help="action bits", type=int, required=True)
parser.add_argument("-lr", help="learning rate", type=float, required=True)
parser.add_argument("-e", help="num epochs", type=int, required=True)
parser.add_argument("-bs", help="batch size", type=int, required=True)
parser.add_argument("-bn", help="batch norm. True (1) or False (0)", type=int, required=True)
parser.add_argument("-dv", help="device", default="cpu")
args = parser.parse_args()


if not os.path.exists(args.s):
    os.makedirs(args.s)

arg_dict = vars(args)
for i, arg in enumerate(arg_dict):
    if i == 0:
        mode = "w"
    else:
        mode = "a"
    print(f"{arg}={arg_dict[arg]}", file=open(os.path.join(args.s, "args.txt"), mode))

BN = True if args.bn == 1 else False

encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.MLP([1024, args.state_bits]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

# encoder_att = torch.nn.MultiheadAttention(embed_dim=args.state_bits, num_heads=4, batch_first=True)
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args.state_bits+args.action_bits, nhead=1, batch_first=True)
decoder_att = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

decoder = blocks.MLP([args.state_bits+args.action_bits, 256, 256, 256, 3], batch_norm=BN)

encoder.to(args.dv)
decoder.to(args.dv)
# encoder_att.to(args.dv)
decoder_att.to(args.dv)

model = DeepSymv3(encoder=encoder, decoder=decoder, decoder_att=decoder_att,
                  device=args.dv, lr=args.lr, path=args.s, coeff=1)
model.print_model()
for name in model.module_names:
    print(f"{name} params={get_parameter_count(getattr(model, name)):,}")

valid_objects = {i: True for i in range(8, 13)}
data = SegmentedSAEFolder(args.d, max_pad=5, valid_objects=valid_objects)
loader = torch.utils.data.DataLoader(data, batch_size=args.bs, num_workers=os.cpu_count())
model.train(args.e, loader)
