import time
import os
import argparse

import torch
import numpy as np
import environment


buffer = []
buffer_type = ""
buffer_lenght = 0


def collect_rollout(env):
    action = 0
    global buffer
    if len(buffer) != 0:
        action = buffer[0]
        buffer = buffer[1:]
    else:
        action = env.sample_random_action()
    position, effect, types = env.step(*action)
    post_position, _ = env.state_obj_poses_and_types()
    return position, types, action, effect, post_position


def populate_buffer(env):
    global buffer_type
    buffer = None
    if buffer_type == "3obj":
        buffer = env.sample_3_objects_moving_together()
    if buffer_type == "mistarget":
        buffer = env.sample_mistarget()
    if buffer_type == "both":
        buffer = env.sample_both()
    if buffer_type == "proximity":
        buffer = env.sample_proximity()

    assert (buffer is not None)

    return buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    parser.add_argument("-b", help="buffer_type", type=str, required=True)
    parser.add_argument("-post", help="post buffer actions", type=int, default=0)
    parser.add_argument("-pre", help="pre buffer actions", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    min_obj = 5 if args.b == "both" else 3
    env = environment.BlocksWorld_v4(gui=0, min_objects=min_obj, max_objects=5)

    np.random.seed()

    # (x, y, z, cos_rx, sin_rx, cos_ry, sin_ry, cos_rz, sin_rz, type)
    states = torch.zeros(args.N, env.max_objects, 10, dtype=torch.float)
    # (obj_i, obj_j, from_x, from_y, to_x, to_y, rot_init, rot_final)
    actions = torch.zeros(args.N, 8, dtype=torch.int)
    # how many objects are there in the scene
    masks = torch.zeros(args.N, dtype=torch.int)
    # (x_f-x_i, y_f-y_i, z_f-z_i,
    #  cos_rx_f-cos_rx_i, sin_rx_f-sin_rx_i,
    #  cos_ry_f-cos_ry_i, sin_ry_f-sin_ry_i,
    #  cos_rz_f-cos_rz_i, sin_rz_f-sin_rz_i)
    # for before picking and after releasing
    effects = torch.zeros(args.N, env.max_objects, 18, dtype=torch.float)
    post_states = torch.zeros(args.N, env.max_objects, 10, dtype=torch.float)

    prog_it = args.N
    buffer_type = args.b
    start = time.time()
    env_it = 0
    i = 0
    buffer = populate_buffer(env)
    buffer_lenght = len(buffer)
    while i < args.N:
        position_pre, obj_types, action, effect, position_post = collect_rollout(env)
        env_it += 1
        if (len(buffer) > args.pre):
            continue
        states[i, :env.num_objects, :-1] = torch.tensor(position_pre, dtype=torch.float)
        states[i, :env.num_objects, -1] = torch.tensor(obj_types, dtype=torch.float)
        actions[i] = torch.tensor(action, dtype=torch.int)
        masks[i] = env.num_objects
        effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)
        post_states[i, :env.num_objects, :-1] = torch.tensor(position_post, dtype=torch.float)
        post_states[i, :env.num_objects, -1] = torch.tensor(obj_types, dtype=torch.float)
        i += 1

        if (env_it) == buffer_lenght + args.post:
            env_it = 0
            env.reset_objects()
            buffer = populate_buffer(env)

        if i % prog_it == 0:
            print(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(masks, os.path.join(args.o, f"mask_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    torch.save(post_states, os.path.join(args.o, f"post_state_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
