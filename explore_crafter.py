import argparse
import os
import shutil

import gym
import crafter
import torch
import numpy as np

parser = argparse.ArgumentParser("Explore the Crafter environment.")
parser.add_argument("-N", help="number of interactions", type=int, required=True)
parser.add_argument("-o", help="output folder", type=str, required=True)
args = parser.parse_args()

if os.path.exists(args.o):
    shutil.rmtree(args.o)
os.makedirs(args.o)

# might have saved them into the folder, and also represent action
# directly as an integer, not a vector. but whatever.
state = torch.zeros(args.N, 3, 64, 64, dtype=torch.uint8)
action = torch.zeros(args.N, 17, dtype=torch.uint8)
effect = torch.zeros(args.N, 3, 64, 64, dtype=torch.int16)
eye = torch.eye(17, dtype=torch.uint8)

env = gym.make("CrafterReward-v1")
it = 0
while it < args.N:
    obs = env.reset()
    obs = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.uint8)
    done = False
    while not done:
        a = env.action_space.sample()
        next_obs, _, _, _ = env.step(a)
        next_obs = torch.tensor(np.transpose(next_obs, (2, 0, 1)), dtype=torch.uint8)
        e = next_obs.short() - obs.short()

        state[it] = obs
        action[it] = eye[a]
        effect[it] = e

        obs = next_obs
        it += 1
        if it == args.N:
            break
        if it % (args.N // 100) == 0:
            print(f"{100*it/args.N}% completed.")

print(f"Collected {args.N} interactions.")
torch.save(state, os.path.join(args.o, "state.pt"))
torch.save(action, os.path.join(args.o, "action.pt"))
torch.save(effect, os.path.join(args.o, "effect.pt"))
