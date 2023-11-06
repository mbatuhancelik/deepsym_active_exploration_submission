import time
import os
import argparse

import torch
import numpy as np
import environment
import copy

from dataset import StateActionEffectDataset
import wandb
import utils
import yaml

import council_manager

action_set = []
action_seperators = []
action_env_space = []


def current_state_batch(env: environment.BlocksworldLightning):
    state , types = env.state_obj_poses_and_types()
    sample = {}
    state = torch.tensor(state)
    types = torch.tensor(types)
    state = torch.cat([state, StateActionEffectDataset.binary[[types.long()]]], dim=-1)
    sample["state"] = state.unsqueeze(0)
    mask = torch.zeros(state.shape[0], dtype=torch.float, device=state.device)
    mask[:env.num_objects] = 1.0
    sample["pad_mask"] = mask
    return sample

def get_action(env: environment.BlocksworldLightning, council, horizon = 0, sample_size = 100):
    sample = current_state_batch(env)
    actions = action_set[:action_seperators[env.num_objects-1], :env.num_objects]
    batch_size = actions.shape[0]
    sample["state"] = sample["state"].repeat((batch_size,1,1))
    sample["pad_mask"] = sample["pad_mask"].repeat((batch_size,1,1))
    sample["action"] = actions
    e = []
    
    for model in council:
    #TODO: BATCH THIS SHIT
        with torch.no_grad():
            _,_, e_pred = model.forward(sample)
        e.append(e_pred.unsqueeze(0))
    
    e = torch.cat(e, dim=0)
    e = e.var(dim=0)
    act = torch.vmap(torch.trace)(e.permute(0,2,1)@e).argmax().item()
    del e
    return action_env_space[act]

    
    
def collect_rollout(env, council):
    action = get_action(env,council)
    position, effect, types = env.step(*action)
    post_position, _ = env.state_obj_poses_and_types()
    return position, types, action, effect, post_position


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-T", help="interaction per episode", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    parser.add_argument("-d", help="accelerator", type=str, required=False, default="cuda")
    parser.add_argument("-id", help="model id", type=str, required=False)
    args = parser.parse_args()

    council = council_manager.load()
    for m in council:
        m.to(args.d)
    if not os.path.exists(args.o):
        try:
            os.makedirs(args.o)
        finally:
            pass


    env = environment.BlocksworldLightning(gui=0, min_objects=5, max_objects=8)
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
    effects = torch.zeros(args.N, env.max_objects, 9, dtype=torch.float)
    post_states = torch.zeros(args.N, env.max_objects, 10, dtype=torch.float)

    prog_it = args.N
    start = time.time()
    env_it = 0
    i = 0
    counter = 0
    for incoming in range(env.max_objects):
        for obj1 in range(incoming+1):
            for dy1 in [-1,0,1]:
                for dy2 in [-1,0,1]:
                    dx1 = 0
                    dx2 = 0
                    rot_before = 1
                    rot_after = 1
                    a = [obj1, incoming, dx1, dy1, dx2, dy2, rot_before, rot_after]
                    action = torch.zeros(env.max_objects, 8, dtype=torch.float)
                    action[a[0], :4] = torch.tensor([1, a[2], a[3], a[6]], dtype=torch.float)
                    action[a[1], 4:] = torch.tensor([1, a[4], a[5], a[7]], dtype=torch.float)
                    action_set.append(action.unsqueeze(0))
                    action_env_space.append(a)
                    counter += 1
        for obj2 in range(incoming):
            for dy1 in [-1,0,1]:
                    for dy2 in [-1,0,1]:
                        dx1 = 0
                        dx2 = 0
                        rot_before = 1
                        rot_after = 1
                        a = [incoming, obj2, dx1, dy1, dx2, dy2, rot_before, rot_after]
                        action = torch.zeros(env.max_objects, 8, dtype=torch.float)
                        action[a[0], :4] = torch.tensor([1, a[2], a[3], a[6]], dtype=torch.float)
                        action[a[1], 4:] = torch.tensor([1, a[4], a[5], a[7]], dtype=torch.float)
                        action_set.append(action.unsqueeze(0))
                        action_env_space.append(a)
                        counter += 1
        action_seperators.append(counter)
    action_set = torch.cat(action_set, dim=0)


    while i < args.N:
        env_it += 1
        position_pre, obj_types, action, effect, position_post = collect_rollout(env, council)
        states[i, :env.num_objects, :-1] = torch.tensor(position_pre, dtype=torch.float)
        states[i, :env.num_objects, -1] = torch.tensor(obj_types, dtype=torch.float)
        actions[i] = torch.tensor(action, dtype=torch.int)
        masks[i] = env.num_objects
        effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)
        post_states[i, :env.num_objects, :-1] = torch.tensor(position_post, dtype=torch.float)
        post_states[i, :env.num_objects, -1] = torch.tensor(obj_types, dtype=torch.float)

        if (env_it) == args.T:
            env_it = 0
            env.reset_objects()

        i += 1
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
