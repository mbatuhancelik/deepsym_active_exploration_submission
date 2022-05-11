import time
import os
import argparse
from PIL import Image

import torch
import numpy as np

import environment

parser = argparse.ArgumentParser("Explore environment.")
parser.add_argument("-N", help="number of interactions", type=int, required=True)
parser.add_argument("-o", help="output folder", type=str, required=True)
parser.add_argument("-i", help="offset index", type=int, required=True)
args = parser.parse_args()

if not os.path.exists(args.o):
    os.makedirs(args.o)


def collect_rollout(env):
    N = len(env.obj_dict)
    rgb_a, depth_a, seg_a = env.state()
    from_idx = np.random.randint(0, N)
    to_idx = np.random.randint(0, N)
    from_obj_id = env.obj_dict[from_idx]
    to_obj_id = env.obj_dict[to_idx]
    from_pos, _ = env._p.getBasePositionAndOrientation(from_obj_id)
    to_pos, _ = env._p.getBasePositionAndOrientation(to_obj_id)
    from_pos = list(from_pos)
    to_pos = list(to_pos)
    to_pos[2] = 0.75
    env.step(from_pos, to_pos)
    rgb_b, depth_b, seg_b = env.state()
    return (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_obj_id, to_obj_id)


env = environment.BlocksWorld(gui=0)
actions = torch.zeros(args.N, 2, dtype=torch.int32)

start = time.time()
for i in range(args.N):
    save_idx = args.i + i
    (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_obj_id, to_obj_id) = collect_rollout(env)

    Image.fromarray(rgb_a).save(os.path.join(args.o, f"{save_idx}_a_rgb.png"))
    torch.save(torch.tensor(depth_a, dtype=torch.float), os.path.join(args.o, f"{save_idx}_a_depth.pt"))
    torch.save(torch.tensor(seg_a, dtype=torch.int32), os.path.join(args.o, f"{save_idx}_a_seg.pt"))

    Image.fromarray(rgb_b).save(os.path.join(args.o, f"{save_idx}_b_rgb.png"))
    torch.save(torch.tensor(depth_b, dtype=torch.float), os.path.join(args.o, f"{save_idx}_b_depth.pt"))
    torch.save(torch.tensor(seg_b, dtype=torch.int32), os.path.join(args.o, f"{save_idx}_b_seg.pt"))
    actions[i, 0], actions[i, 1] = from_obj_id, to_obj_id
    env.reset_objects()

torch.save(actions, os.path.join(args.o, "action.pt"))
end = time.time()
print(f"Completed in {end-start:.2f} seconds.")
