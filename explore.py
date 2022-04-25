import time
import os
import shutil
import argparse
from PIL import Image

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
    to_pos = list(to_pos)
    to_pos[2] = 0.75
    env.step(list(from_pos), list(to_pos))
    rgb_b, depth_b, seg_b = env.state()
    return (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b)


env = environment.BlocksWorld(gui=1)

start = time.time()
for i in range(args.N):
    print(i)
    (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b) = collect_rollout(env)

    Image.fromarray(rgb_a).save(os.path.join(args.o, f"{i}_a_rgb.png"))
    np.save(os.path.join(args.o, f"{i}_a_depth.npy"), depth_a)
    np.save(os.path.join(args.o, f"{i}_a_seg.npy"), seg_a)

    Image.fromarray(rgb_b).save(os.path.join(args.o, f"{i}_b_rgb.png"))
    np.save(os.path.join(args.o, f"{i}_b_depth.npy"), depth_b)
    np.save(os.path.join(args.o, f"{i}_b_seg.npy"), seg_b)

end = time.time()
print(f"Completed in {end-start:.2f} seconds.")
