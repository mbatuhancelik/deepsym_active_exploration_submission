import time
import os
import shutil
import argparse
from PIL import Image

import torch
import numpy as np

import environment

parser = argparse.ArgumentParser("Explore environment.")
parser.add_argument("-N", help="number of interactions", type=int, required=True)
parser.add_argument("-o", help="output folder", type=str, required=True)
args = parser.parse_args()

if os.path.exists(args.o):
    shutil.rmtree(args.o)
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
    return (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_pos, to_pos)


env = environment.BlocksWorld(gui=1)
actions = torch.zeros(args.N, 6, dtype=torch.float)

start = time.time()
for i in range(args.N):
    print(i)
    (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_pos, to_pos) = collect_rollout(env)

    Image.fromarray(rgb_a).save(os.path.join(args.o, f"{i}_a_rgb.png"))
    np.save(os.path.join(args.o, f"{i}_a_depth.npy"), depth_a)
    torch.save(torch.tensor(seg_a, dtype=torch.int32), os.path.join(args.o, f"{i}_a_seg.pt"))
    # np.save(os.path.join(args.o, f"{i}_a_seg.npy"), seg_a)

    Image.fromarray(rgb_b).save(os.path.join(args.o, f"{i}_b_rgb.png"))
    np.save(os.path.join(args.o, f"{i}_b_depth.npy"), depth_b)
    torch.save(torch.tensor(seg_b, dtype=torch.int32), os.path.join(args.o, f"{i}_b_seg.pt"))
    # np.save(os.path.join(args.o, f"{i}_b_seg.npy"), seg_b)

    actions[i, 0], actions[i, 1], actions[i, 2] = from_pos[0], from_pos[1], from_pos[2]
    actions[i, 3], actions[i, 4], actions[i, 5] = to_pos[0], to_pos[1], to_pos[2]

torch.save(actions, os.path.join(args.o, "action.pt"))
print("%d" % args.N, file=open(os.path.join(args.o, "info.txt"), "w"))
end = time.time()
print(f"Completed in {end-start:.2f} seconds.")
