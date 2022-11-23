import time
import os
import argparse

import torch
import numpy as np

import environment
import utils


def collect_rollout(env):
    position_a , types = env.state()
    action = env.sample_random_action()
    effect, _ = env.step(*action)
    position_b , _ = env.state()
    return position_a, position_b ,types ,  action, effect


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    env = environment.BlocksWorld_v4(gui=0, min_objects=8, max_objects=13)
    # env = environment.BlocksWorld_v3(gui=0, min_objects=5, max_objects=9)
    np.random.seed()

    #TODO: add mask
    states = torch.zeros(args.N, 1, env.max_objects,  6, dtype=torch.float)
    actions = torch.zeros(args.N, 6, dtype=torch.int32)
    masks = torch.zeros(args.N , dtype=torch.int)
    effects = torch.zeros(args.N, env.max_objects, 6, dtype=torch.float)
    types = torch.zeros(args.N, env.max_objects, dtype=torch.float)

    post_states = torch.zeros(args.N, env.max_objects, 6, dtype=torch.float)
    
    prog_it = args.N // 2
    start = time.time()
    env_it = 0
    i = 0
    non_existing_pos = [[0 for i in range(6)]]
    while i < args.N:
        print(f"hello{i}")
        env_it += 1
        position_pre, position_after, obj_types , action , effect = collect_rollout(env)
        from_idx, to_idx , rotation_before, rotation_after = action
        mask_index = env.max_objects
        while len(position_after) < env.max_objects:
            mask_index -= 1
            position_after = np.concatenate((position_after,non_existing_pos))
            position_pre = np.concatenate((position_pre,non_existing_pos))
            obj_types = np.concatenate((obj_types, [0]))
        masks[i] = mask_index
        states[i] = torch.tensor(position_pre, dtype=torch.float)
        post_states[i] = torch.tensor(position_after, dtype=torch.float)
        actions[i]= torch.tensor([from_idx[0],from_idx[1], to_idx[0], to_idx[1] , rotation_before , rotation_after])
        types[i] = torch.tensor(obj_types)
        if (env_it) == 2:
            env_it = 0
            env.reset_objects()

        i += 1
        if i % prog_it == 0:
            print(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(post_states, os.path.join(args.o, f"post_state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    torch.save(types, os.path.join(args.o, f"segmentation_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
