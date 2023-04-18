import argparse

import torch
import os


parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-s", "--small", help="smaller dataset", type=str, required=True)
parser.add_argument("-l", "--large", help="appended dataset", type=str, required=True)
parser.add_argument("-o", help="output folder", type=str, required=True)
args = parser.parse_args()

path = os.path.join("data", args.large)
l_state = torch.load(os.path.join(path, "state.pt"))
l_action = torch.load(os.path.join(path, "action.pt"))
l_effect = torch.load(os.path.join(path, "effect.pt"))
l_mask = torch.load(os.path.join(path, "mask.pt"))


path = os.path.join("data", args.small)
s_state = torch.load(os.path.join(path, "state.pt"))
s_action = torch.load(os.path.join(path, "action.pt"))
s_effect = torch.load(os.path.join(path, "effect.pt"))
s_mask = torch.load(os.path.join(path, "mask.pt"))

len_small = s_state.shape[0]
len_large = l_state.shape[0]

divide_index = int(len_small/2)
action = torch.concat([s_action[:divide_index], l_action[:len_small], s_action[divide_index+1:]] , dim=0)
state = torch.concat([s_state[:divide_index], l_state[:len_small], s_state[divide_index+1:]], dim=0)
effect = torch.concat([s_effect[:divide_index], l_effect[:len_small], s_effect[divide_index+1:]], dim=0)
mask = torch.concat([s_mask[:divide_index], l_mask[:len_small], s_mask[divide_index+1:]], dim=0)

shuffle = torch.randperm(action.size()[0])

action=action[shuffle]
state=state[shuffle]
effect=effect[shuffle]
mask=mask[shuffle]
if not os.path.exists(args.o):
        os.makedirs(args.o)

torch.save(state, os.path.join(args.o, f"state.pt"))
torch.save(action, os.path.join(args.o, f"action.pt"))
torch.save(mask, os.path.join(args.o, f"mask.pt"))
torch.save(effect, os.path.join(args.o, f"effect.pt"))

