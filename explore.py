import time
import os
import argparse

import torch
import numpy as np

import environment


def normalize_depth_img(img):
    min_val = 0.0
    max_val = 1.0
    return (((img - min_val) / (max_val - min_val))*255).astype(np.uint8)

def collect_rollout(env):
    rgb_a, depth_a, seg_a = env.state()
    depth_a = normalize_depth_img(depth_a)
    action = env.sample_random_action()
    effect = env.step(*action)
    return (rgb_a, depth_a, seg_a), action, effect


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    env = environment.BlocksWorld_v2(gui=0, min_objects=1, max_objects=3)
    np.random.seed()

    states = torch.zeros(args.N, 1, 256, 256, dtype=torch.uint8)
    segmentations = torch.zeros(args.N, 256, 256, dtype=torch.uint8)
    actions = torch.zeros(args.N, 2, dtype=torch.int32)
    effects = torch.zeros(args.N, env.max_objects, 7, dtype=torch.float)

    prog_it = args.N // 20
    start = time.time()
    env_it = 0
    i = 0
    while i < args.N:
        env_it += 1
        (rgb_a, depth_a, seg_a), (from_idx, to_idx), effect = collect_rollout(env)
        if seg_a.max() < 4:
            env_it = 0
            env.reset_objects()
            continue

        states[i, 0] = torch.tensor(depth_a, dtype=torch.uint8)
        segmentations[i] = torch.tensor(seg_a, dtype=torch.uint8)
        actions[i, 0], actions[i, 1] = from_idx, to_idx
        effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)
        if (env_it) == env.num_objects*5:
            env_it = 0
            env.reset_objects()

        i += 1
        if i % prog_it == 0:
            print(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    torch.save(segmentations, os.path.join(args.o, f"segmentation_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
