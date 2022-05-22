import argparse
import os

import torch

import environment

parser = argparse.ArgumentParser("Explore the MNIST N-puzzle environment.")
parser.add_argument("-N", help="number of interactions", type=int, required=True)
parser.add_argument("-o", help="output folder", type=str, required=True)
parser.add_argument("-i", help="index", type=int, required=True)
parser.add_argument("-s", help="size", type=int, required=True)
args = parser.parse_args()

if not os.path.exists(args.o):
    os.makedirs(args.o)

height = args.s*28
width = args.s*28

state = torch.zeros(args.N, 3, height, width, dtype=torch.uint8)
action = torch.zeros(args.N, 4, dtype=torch.uint8)
effect = torch.zeros(args.N, 3, height, width, dtype=torch.int16)
eye = torch.eye(4, dtype=torch.uint8)

env = environment.TilePuzzleMNIST(size=args.s, random=True)
it = 0
while it < args.N:
    obs = env.reset()
    a = torch.randint(4, ()).item()
    next_obs = env.step(a)
    e = next_obs - obs

    state[it] = (obs * 255).byte()
    action[it] = eye[a]
    effect[it] = (e * 255).short()
    it += 1

torch.save(state, os.path.join(args.o, f"state{args.i}.pt"))
torch.save(action, os.path.join(args.o, f"action{args.i}.pt"))
torch.save(effect, os.path.join(args.o, f"effect{args.i}.pt"))
