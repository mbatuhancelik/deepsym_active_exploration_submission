import time
import os
import argparse

import torch
import numpy as np

import environment


def collect_rollout(env):
    # N = len(env.obj_dict)
    N = 6
    rgb_a, depth_a, seg_a = env.state()
    from_idx = np.random.randint(0, N)
    to_idx = np.random.randint(0, N)
    # from_obj_id = env.obj_dict[from_idx]
    # to_obj_id = env.obj_dict[to_idx]
    # effect = env.step(from_obj_id, to_obj_id)
    effect = env.step(from_idx, to_idx)
    # rgb_b, depth_b, seg_b = env.state()
    # return (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_idx, to_idx), effect
    return (rgb_a, depth_a, seg_a), (from_idx, to_idx), effect


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    env = environment.BlocksWorld_v2(gui=1, min_objects=3, max_objects=3)
    # env.reset_object_poses()
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
        # (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_obj_id, to_obj_id), effect = collect_rollout(env)
        (rgb_a, depth_a, seg_a), (from_idx, to_idx), effect = collect_rollout(env)
        # if seg_a.max() < 8:
        #     continue

        depth_a = (((depth_a - depth_a.min()) / (depth_a.max() - depth_a.min()))*255).astype(np.uint8)
        # states[i, :3] = torch.tensor(np.transpose(rgb_a, (2, 0, 1)), dtype=torch.uint8)
        states[i, 0] = torch.tensor(depth_a, dtype=torch.uint8)
        segmentations[i] = torch.tensor(seg_a, dtype=torch.uint8)
        actions[i, 0], actions[i, 1] = from_idx, to_idx
        effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)
        if (env_it) == 20:
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
