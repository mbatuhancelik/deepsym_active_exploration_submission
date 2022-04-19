"""Train DeepSym"""
import argparse
import os
import subprocess

import torch

from models import DeepSymbolGenerator
from dataset import StateActionDataset
import blocks

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-s", help="save folder", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.s):
    os.makedirs(args.s)

BN = True
ACTION_BITS = 13
STATE_BITS = 50
NUM_EPOCH = 30
LR = 0.0001
BATCH_SIZE = 128

encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.MLP([1024, STATE_BITS]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

decoder = torch.nn.Sequential(
    blocks.MLP([STATE_BITS+ACTION_BITS, 1024]),
    blocks.Reshape([-1, 1024, 1, 1]),
    blocks.ConvTransposeBlock(in_channels=1024, out_channels=512, kernel_size=4, stride=1, padding=0, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1)
)

encoder.to("cuda")
decoder.to("cuda")

model = DeepSymbolGenerator(encoder=encoder, decoder=decoder, subnetworks=[],
                            device="cuda", lr=1e-4, path=args.s, coeff=9.0)
model.print_model()

# collect data
for e in range(NUM_EPOCH):
    subprocess.run(["python", "explore.py", "-N", "10000"])
    data = StateActionDataset("./data")
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=12)
    model.train(1, loader)
