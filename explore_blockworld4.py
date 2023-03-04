import time
import os
import argparse

import torch
import numpy as np
import environment
from tqdm import tqdm 


buffer = []

def collect_rollout(env):
    _ , types = env.state()
    action = 0
    global buffer
    if len(buffer) == 0:
        action = buffer[0]
        buffer = buffer[1:]
    else:    
        action = env.sample_random_action()
    position_a, effect, _ = env.step(*action)
    position_b, _ = env.state()
    return position_a, position_b, types, action, effect
def populate_buffer(env):
    if np.random.rand(1)[0] < 0.5:
        return env.sample_3_objects_moving_together()
    else:
        return env.sample_ungrappable()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-T", help="interaction per episode", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    env = environment.BlocksWorld_v4(gui=0, min_objects=8, max_objects=13)
    np.random.seed()

    # (x, y, z, rx, ry, rz, type)
    states = torch.zeros(args.N, env.max_objects, 10, dtype=torch.float)
    # (from_x, from_y, to_x, to_y, rot_init, rot_final)
    actions = torch.zeros(args.N, 8, dtype=torch.int32)
    # how many objects are there in the scene
    masks = torch.zeros(args.N, dtype=torch.int)
    # (x_f-x_i, y_f-y_i, z_f-z_i, rx_f-rx_i, ry_f-ry_i, rz_f-rz_i)
    effects = torch.zeros(args.N, env.max_objects, 18, dtype=torch.float)

    prog_it = args.N // 100
    start = time.time()
    env_it = 0
    i = 0
    buffer = populate_buffer(env)
    while i < args.N:
        env_it += 1
        position_pre, position_after, obj_types, action, effect = collect_rollout(env)
        source, target, dx1, dy1, dx2, dy2, pre_rot, after_rot = action
        states[i, :env.num_objects, :9] = torch.tensor(position_pre, dtype=torch.float)
        states[i, :env.num_objects, 9] = torch.tensor(obj_types, dtype=torch.float)
        actions[i] = torch.tensor([source, target,dx1,dy1,dx2 ,dy2, pre_rot, after_rot])
        masks[i] = env.num_objects
        effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)

        if (env_it) == args.T:
            env_it = 0
            env.reset_objects()
            buffer = populate_buffer(env)

        i += 1
        if i % prog_it == 0:
            print(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(masks, os.path.join(args.o, f"mask_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
