"""Train DeepSym"""
import argparse
import os

import torch

import blocks
from models import DeepSymv4
from dataset import CrafterDataset
from utils import get_parameter_count

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-s", help="save folder", type=str, required=True)
parser.add_argument("-d", help="data folder", type=str, required=True)
parser.add_argument("-state_bits", help="state bits", type=int, required=True)
parser.add_argument("-lr", help="learning rate", type=float, required=True)
parser.add_argument("-e", help="num epochs", type=int, required=True)
parser.add_argument("-bs", help="batch size", type=int, required=True)
parser.add_argument("-bn", help="batch norm. True (1) or False (0)", type=int, required=True)
parser.add_argument("-dv", help="device", default="cpu")
args = parser.parse_args()

ACTION_BITS = 17

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
    blocks.Reshape([-1, 37]),
    blocks.MLP([37, 128, 128, 128, args.state_bits]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0),
    blocks.Reshape([-1, 63, args.state_bits]),
)

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args.state_bits+ACTION_BITS, nhead=1, batch_first=True)
decoder_att = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

decoder = blocks.MLP([args.state_bits+ACTION_BITS, 256, 256, 256, 37], batch_norm=BN)

encoder.to(args.dv)
decoder.to(args.dv)
decoder_att.to(args.dv)

model = DeepSymv4(encoder=encoder, decoder=decoder, decoder_att=decoder_att,
                  device=args.dv, lr=args.lr, path=args.s, coeff=1)
model.print_model()
for name in model.module_names:
    print(f"{name} params={get_parameter_count(getattr(model, name)):,}")

data = CrafterDataset(args.d)
loader = torch.utils.data.DataLoader(data, batch_size=args.bs, num_workers=os.cpu_count())
model.train(args.e, loader)
