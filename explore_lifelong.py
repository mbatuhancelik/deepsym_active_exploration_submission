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
buffer = []


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

def current_encodings(env, model):
    sample = current_state_batch(env)
    encode_state(model, sample)
    sample["state"] = sample["state"].repeat((sample["state"].shape[1],1,1))
    sample["encoding"] = sample["encoding"].repeat((sample["state"].shape[1],1,1))
    sample["heads"] = sample["heads"].repeat((sample["state"].shape[1],1,1))
    objs = torch.arange(sample["state"].shape[1])

    obj_encodings = get_sample_obj_graph(sample, objs)
    return obj_encodings

def collect_rollout(env, model, action_history):
    obj_encodings = current_encodings(env,model)
    interaction_map = {}
    for i, obj1 in enumerate(obj_encodings):
        for k, obj2 in enumerate(obj_encodings):
            obj = (obj1 , obj2)
            if obj not in action_history.keys():
                action_history[obj] = 0
            interaction_map[action_history[obj]] = (i,k)
    int_list = list(interaction_map.keys())
    int_list.sort()


    
    obj_1, obj_2 = interaction_map[int_list[0]]
    action_history[(obj_encodings[obj_1], obj_encodings[obj_2])] += 1
    
    action = env.sample_random_action()
    action[0] = obj_1
    action[1] = obj_2
    
    position, effect, types = env.step(*action)
    post_position, _ = env.state_obj_poses_and_types()
    return position, types, action, effect, post_position

def get_heads_tuple(head , indices, vals):
    x = head[indices]
    x[vals.count_nonzero():] = 0
    x = x.permute(1, 0)[indices]
    x[vals.count_nonzero():] = 0
    x = x.permute(1,0)
    x = x.reshape(-1)
    return x[x.nonzero()].reshape(-1).tolist()
samples = []
action_set = set()
def single_object_map_filter(sample, obj):
    batch_size = sample["state"].shape[0]
    locals = sample["state"][torch.arange(batch_size), obj.squeeze(), :2]
    dist_obj = (sample["state"][:,: ,:2] - locals.unsqueeze(1)) ** 2
    rel_obj = copy.deepcopy(dist_obj.sum(dim=-1) < 0.075**2).int()
    obj_vals = (sample["encoding"].cpu() * rel_obj.int().unsqueeze(-1) * (2 ** torch.arange(4)).unsqueeze(0)).sum(dim = -1)
    obj_vals[ torch.arange(batch_size) , obj] += 16
    return  torch.sort(obj_vals, descending=True)
def get_sample_obj_graph(sample, objs):
    obj_encodings = []
    vals1, indices1  = single_object_map_filter(sample, objs)
    for i in range(sample["state"].shape[0]):
        heads1 = get_heads_tuple(sample["heads"][i], indices1[i], vals1[i])
        obj_vals1 = vals1[i].reshape(-1)
        obj_vals1 = obj_vals1[obj_vals1.nonzero()].reshape(-1).tolist()
        obj_encodings.append((tuple(obj_vals1), tuple(heads1)))
    return obj_encodings
def encode_state(model, sample):
    with torch.no_grad():
            attn_weights = model.attn_weights(sample["state"], sample["pad_mask"], eval_mode=True)
            encoding = model.encode(sample["state"], eval_mode=True)
            attn_weights = (attn_weights * (2 ** torch.arange(4)).reshape(1,4,1,1).to(attn_weights.device)).sum(dim=1)
    sample["heads"] = copy.deepcopy(attn_weights)
    sample["encoding"] = copy.deepcopy(encoding)
    return encoding, attn_weights
def get_sample_action_set(model, sample):
    action_set = []
    encoding , attn_weights = encode_state(model, sample)
    batch_size = sample["action"].shape[0]
    sample["objs"] = sample["action"][:, :,[0,4]].argmax(dim=1)
    vals1, indices1  = single_object_map_filter(sample, sample["objs"][:, 0])
    vals2, indices2  = single_object_map_filter(sample, sample["objs"][:, 1])
    sample["action"]= sample["action"].sum(dim=-2)
    for i in range(batch_size):
        heads1 = get_heads_tuple(attn_weights[i], indices1[i], vals1[i])
        heads2 = get_heads_tuple(attn_weights[i], indices2[i], vals2[i])
        obj_vals1 = vals1[i].reshape(-1)
        obj_vals1 = obj_vals1[obj_vals1.nonzero()].reshape(-1).tolist()
        
        obj_vals2 = vals2[i].reshape(-1)
        obj_vals2 = obj_vals2[obj_vals2.nonzero()].reshape(-1).tolist()
        
        key = [(tuple(obj_vals1), tuple(heads1)), (tuple(obj_vals2),tuple(heads2)), (tuple(sample["action"][i].reshape(-1).tolist()))]
        action_set.append(key)
    return action_set
def get_dataset_action_set(model):
    loader = torch.utils.data.DataLoader(dataset, 2048, shuffle=False)
    action_set = {}
    for sample in iter(loader):
        action_encodings = (get_sample_action_set(model, sample))
        for act in action_encodings:
            source = act[0]
            target = act[1]
            action = act[2]
            key = (act[0], act[1])
            if key not in action_set.keys():
                action_set[key] = 0
            action_set[key] += 1
    return action_set
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-T", help="interaction per episode", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    parser.add_argument("-id", help="model id", type=str, required=True)
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = torch.load("model.pt")
    
    dataset = StateActionEffectDataset(config["dataset_name"], split="")
    if not os.path.exists(args.o):
        os.makedirs(args.o)


    env = environment.BlocksworldLightning(gui=0, min_objects=5, max_objects=8)
    np.random.seed()
    
    action_set = get_dataset_action_set(model)
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

    while i < args.N:
        env_it += 1
        position_pre, obj_types, action, effect, position_post = collect_rollout(env, model, action_set)
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
