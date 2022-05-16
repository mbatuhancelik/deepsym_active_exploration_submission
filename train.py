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
    blocks.ConvBlock(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=512, out_channels=args.state_bits, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

encoder_att = torch.nn.MultiheadAttention(embed_dim=args.state_bits, num_heads=4, batch_first=True)
decoder_att = torch.nn.MultiheadAttention(embed_dim=args.state_bits+args.action_bits, num_heads=1, batch_first=True)

decoder = blocks.MLP([args.state_bits+args.action_bits, 128, 128, 128, 7], batch_norm=BN)

encoder.to(device)
decoder.to(device)
encoder_att.to(device)
decoder_att.to(device)

model = DeepSymv3(encoder=encoder, decoder=decoder, encoder_att=encoder_att, decoder_att=decoder_att,
                  device=device, lr=args.lr, path=args.s, coeff=1.0)
model.print_model()
encoder_param_count = get_parameter_count(model.encoder)
encoder_att_param_count = get_parameter_count(model.encoder_att)
decoder_param_count = get_parameter_count(model.decoder)
decoder_att_param_count = get_parameter_count(model.decoder_att)
print(f"Encoder params={encoder_param_count:,}")
print(f"Encoder att. params={encoder_att_param_count:,}")
print(f"Decoder params={decoder_param_count:,}")
print(f"Decoder att. params={decoder_att_param_count:,}")
print(f"Total={encoder_param_count+encoder_att_param_count+decoder_param_count+decoder_att_param_count:,}")

valid_objects = {i: True for i in range(8, 18)}
data = SegmentedSAEFolder(args.d, max_pad=10, valid_objects=valid_objects)
loader = torch.utils.data.DataLoader(data, batch_size=args.bs, num_workers=os.cpu_count())
model.train(args.e, loader)
