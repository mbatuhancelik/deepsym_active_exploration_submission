import time
import os
import argparse

import torch
import numpy as np

import environment


def normalize_depth_img(img):
    min_val = 0.9958513
    max_val = 0.9977687
    return (((img - min_val) / (max_val - min_val))*255).astype(np.uint8)


def collect_rollout(env):
    _, depth_a, _ = env.state()
    action = env.sample_random_action()
    env.step(*action)
    _, depth_b, _ = env.state()
    depth_a = normalize_depth_img(depth_a)
    depth_b = normalize_depth_img(depth_b)
    return depth_a, action, depth_b


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
    actions = torch.zeros(args.N, 2, dtype=torch.int32)
    effects = torch.zeros(args.N, 1, 256, 256, dtype=torch.uint8)

    prog_it = args.N // 20
    start = time.time()
    env_it = 0
    i = 0
    while i < args.N:
        env_it += 1
        depth_a, (from_idx, to_idx), depth_b = collect_rollout(env)

        states[i, 0] = torch.tensor(depth_a, dtype=torch.uint8)
        actions[i, 0], actions[i, 1] = from_idx, to_idx
        effects[i, 0] = torch.tensor(depth_b, dtype=torch.uint8)
        if (env_it) == env.num_objects*5:
            env_it = 0
            env.reset_objects()

        i += 1
        if i % prog_it == 0:
            print(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
