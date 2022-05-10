"""Train DeepSym"""
import argparse
import os
import subprocess

import torch

import blocks
from models import DeepSymbolGenerator
from dataset import StateActionDataset, StateActionEffectDataset
from utils import get_parameter_count

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-s", help="save folder", type=str, required=True)
parser.add_argument("-d", help="data folder", type=str, required=True)
parser.add_argument("-dt", help="dataset type. StateAction (0) or StateActionEffect (1)", type=int, required=True)
parser.add_argument("-state_bits", help="state bits", type=int, required=True)
parser.add_argument("-action_bits", help="action bits", type=int, required=True)
parser.add_argument("-lr", help="learning rate", type=float, required=True)
parser.add_argument("-e", help="num epochs", type=int, required=True)
parser.add_argument("-bs", help="batch size", type=int, required=True)
parser.add_argument("-bn", help="batch norm. True (1) or False (0)", type=int, required=True)
args = parser.parse_args()


if not os.path.exists(args.s):
    os.makedirs(args.s)

arg_dict = vars(args)
for arg in arg_dict:
    print(f"{arg}={arg_dict[arg]}", file=open(os.path.join(args.s, "args.txt"), "a"))

device = "cuda" if torch.cuda.is_available() else "cpu"

BN = True if args.bn == 1 else False

encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=512, out_channels=args.state_bits, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

decoder = torch.nn.Sequential(
    # blocks.MLP([args.state_bits+args.action_bits, 512]),
    blocks.Reshape([-1, args.state_bits+args.action_bits, 1, 1]),
    blocks.ConvTransposeBlock(in_channels=args.state_bits+args.action_bits, out_channels=512, kernel_size=4, stride=1, padding=0, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
)

encoder.to(device)
decoder.to(device)

model = DeepSymbolGenerator(encoder=encoder, decoder=decoder, device=device, lr=args.lr, path=args.s, coeff=81.0)
model.print_model()
encoder_param_count = get_parameter_count(model.encoder)
decoder_param_count = get_parameter_count(model.decoder)
print(f"Encoder params={encoder_param_count:,}")
print(f"Decoder params={decoder_param_count:,}")
print(f"Total={encoder_param_count+decoder_param_count:,}")

# collect data
for e in range(args.e):
    subprocess.run(["python", "explore_crafter.py", "-N", "1024", "-o", args.d])

    if args.dt == 0:
        data = StateActionDataset(args.d)
    else:
        data = StateActionEffectDataset(args.d)

    loader = torch.utils.data.DataLoader(data, batch_size=args.bs, num_workers=12)
    model.train(1, loader)
