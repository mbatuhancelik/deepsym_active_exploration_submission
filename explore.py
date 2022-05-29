import time
import os
import argparse

import torch
import numpy as np

import environment


def collect_rollout(env):
    N = len(env.obj_dict)
    rgb_a, depth_a, seg_a = env.state()
    from_idx = np.random.randint(0, N)
    to_idx = np.random.randint(0, N)
    from_obj_id = env.obj_dict[from_idx]
    to_obj_id = env.obj_dict[to_idx]
    effect = env.step(from_obj_id, to_obj_id)
    rgb_b, depth_b, seg_b = env.state()
    return (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_idx, to_idx), effect


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    env = environment.BlocksWorld(gui=1, min_objects=1, max_objects=5)
    # env.reset_object_poses()
    np.random.seed()

    states = torch.zeros(args.N, 1, 256, 256, dtype=torch.uint8)
    segmentations = torch.zeros(args.N, 256, 256, dtype=torch.uint8)
    actions = torch.zeros(args.N, 2, dtype=torch.int32)
    effects = torch.zeros(args.N, env.max_objects, 7, dtype=torch.float)

    prog_it = args.N // 20
    start = time.time()
    env_it = 0
    for i in range(args.N):
        env_it += 1
        save_idx = args.i + i
        (rgb_a, depth_a, seg_a), (rgb_b, depth_b, seg_b), (from_obj_id, to_obj_id), effect = collect_rollout(env)

        depth_a = (((depth_a - depth_a.min()) / (depth_a.max() - depth_a.min()))*255).astype(np.uint8)
        # states[i, :3] = torch.tensor(np.transpose(rgb_a, (2, 0, 1)), dtype=torch.uint8)
        states[i, 0] = torch.tensor(depth_a, dtype=torch.uint8)
        segmentations[i] = torch.tensor(seg_a, dtype=torch.uint8)
        actions[i, 0], actions[i, 1] = from_obj_id, to_obj_id
        effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)
        if (env_it) == len(env.obj_dict):
            env_it = 0
            env.reset_objects()

        if (i+1) % prog_it == 0:
            print(f"Proc {args.i}: {100*(i+1)/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    torch.save(segmentations, os.path.join(args.o, f"segmentation_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
