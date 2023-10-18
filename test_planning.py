import os
import pickle
import argparse

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

import models
import utils
import mcts
import environment


def random_action():
    obj1 = np.random.randint(0, 5)
    dx1 = np.random.randint(-1, 2)
    obj2 = np.random.randint(0, 5)
    dx2 = np.random.randint(-1, 2)
    return f"{obj1},0,{dx1},{obj2},0,{dx2}"


parser = argparse.ArgumentParser("Test planning")
parser.add_argument("-i", help="Wandb run id", type=str)
args = parser.parse_args()

run = wandb.init(entity="colorslab", project="active_exploration", resume="must", id=args.i)
wandb.config.update({"device": "cpu"})

model = utils.create_model_from_config(dict(run.config))
model.load("_best", from_wandb=True)
model.print_model()
for name in model.module_names:
    module = getattr(model, name)
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
subsymbolic_forward = mcts.SubsymbolicForwardModel(model)

# sym_model = models.SymbolForward(input_dim=run.config["latent_dim"]+run.config["action_dim"],
#                                  hidden_dim=run.config["forward_model"]["hidden_unit"],
#                                  output_dim=run.config["latent_dim"],
#                                  num_layers=run.config["forward_model"]["layer"],
#                                  num_heads=run.config["n_attention_heads"])
path = os.path.join(run.config["save_folder"], "symbol_forward.pt")
module_dict = torch.load(wandb.restore(path).name)
# sym_model.load_state_dict(module_dict)
# for p in sym_model.parameters():
#     p.requires_grad = False
# symbolic_forward = mcts.SymbolicForwardModel(sym_model)

# dict_file = wandb.restore(os.path.join(run.config["save_folder"], "transition.pkl"),
#                           run_path=run.path).name
# dict_forward = pickle.load(open(dict_file, "rb"))

one_hot = torch.tensor([[0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]])
env = environment.BlocksWorld_v4(gui=0, min_objects=5, max_objects=5)

img_out_path = "out/imgs"
if not os.path.exists(img_out_path):
    os.makedirs(img_out_path)

print()
for n_action in range(1, 6):
    subsymbolic_success = 0
    symbolic_success = 0
    dict_success = 0
    dict_found = 0
    for i in range(100):
        if not os.path.exists(os.path.join(img_out_path, str(n_action), str(i))):
            os.makedirs(os.path.join(img_out_path, str(n_action), str(i)))

        np.random.seed(i+200)
        env.reset()

        actions = []
        for _ in range(n_action):
            actions.append(random_action())

        # initial state
        poses, types = env.state()
        state = torch.tensor(np.hstack([poses, types.reshape(-1, 1)]))
        init_state = torch.cat([state[:, :-1], one_hot[[state[:, -1].long()]]], dim=-1)
        with torch.no_grad():
            mask = torch.ones(1, init_state.shape[0])
            init_obj = model.encode(init_state.unsqueeze(0), eval_mode=True)
            init_rel = model.attn_weights(init_state.unsqueeze(0), mask, eval_mode=True)
        init_str = utils.to_str_state(init_obj[0], init_rel[0])

        # act
        print("Actions:", file=open(os.path.join(img_out_path, str(n_action), str(i), "actions.txt"), "w"))
        for j, action in enumerate(actions):
            print(action, file=open(os.path.join(img_out_path, str(n_action), str(i), "actions.txt"), "a"))
            action = [int(x) for x in action.split(",")]
            _, _, _, images = env.step(action[0], action[3], action[1], action[2], action[4], action[5],
                                       grap_angle=1, put_angle=1, get_images=True)
            for k, img in enumerate(images):
                plt.imshow(img)
                plt.savefig(os.path.join(img_out_path, str(n_action), str(i), f"{j}_{k}.png"))

        # final state
        poses, types = env.state()
        print(poses[:, :3], types, file=open(os.path.join(img_out_path, str(n_action), str(i), "objects.txt"), "w"))
        state = torch.tensor(np.hstack([poses, types.reshape(-1, 1)]))
        goal_state = torch.cat([state[:, :-1], one_hot[[state[:, -1].long()]]], dim=-1)
        with torch.no_grad():
            mask = torch.ones(1, goal_state.shape[0])
            goal_obj = model.encode(goal_state.unsqueeze(0), eval_mode=True)
            goal_rel = model.attn_weights(goal_state.unsqueeze(0), mask, eval_mode=True)
        goal_str = utils.to_str_state(goal_obj[0], goal_rel[0])

        subsym_init = mcts.SubsymbolicState(init_state, goal_state)
        # sym_init = mcts.SymbolicState(init_str, goal_str)
        last_obj = [tuple(utils.binary_tensor_to_str(init_obj[0]))]
        last_rel = []
        for rel in init_rel[0]:
            last_rel.append(tuple(utils.binary_tensor_to_str(rel)))
        last_rel = [tuple(last_rel)]
        not_found = False
        for action in actions:
            # subsymbolic
            subsym_init = subsymbolic_forward(subsym_init, action)
            # symbolic
            # sym_init = symbolic_forward(sym_init, action)

            # # dict
            # act = torch.zeros(5, 8, dtype=torch.float)
            # act_str = action.split(",")
            # act[int(act_str[0]), :4] = torch.tensor([1, float(act_str[1]), float(act_str[2]), 1])
            # act[int(act_str[3]), 4:] = torch.tensor([1, float(act_str[4]), float(act_str[5]), 1])
            # act = tuple(utils.binary_tensor_to_str(act))

            # new_obj = []
            # new_rel = []
            # key_found = False
            # for last_obj_i, last_rel_i in zip(last_obj, last_rel):
            #     permuted_ary = utils.permute(list(range(len(last_obj_i))))
            #     for perm in permuted_ary:
            #         last_obj_i_perm = [last_obj_i[j] for j in perm]
            #         last_rel_i_perm = []
            #         for rel in last_rel_i:
            #             row_permuted = [rel[j] for j in perm]
            #             for row in range(len(row_permuted)):
            #                 row_permuted[row] = "".join([row_permuted[row][j] for j in perm])
            #             last_rel_i_perm.append(tuple(row_permuted))
            #         key = (tuple(last_obj_i_perm), tuple(last_rel_i_perm))
            #         if key in dict_forward:
            #             key_found = True
            #             act = [act[j] for j in perm]
            #             act = tuple(act)
            #             break

            #     if key_found:
            #         if key in dict_forward:
            #             if act in dict_forward[key]:
            #                 outcomes = dict_forward[key][act]
            #                 for out_key in outcomes:
            #                     back_perm = [perm.index(j) for j in range(len(perm))]
            #                     new_obj.append(tuple([out_key[0][j] for j in back_perm]))
            #                     permuted_rels = []
            #                     for rel in out_key[1]:
            #                         row_permuted = [rel[j] for j in back_perm]
            #                         for row in range(len(row_permuted)):
            #                             row_permuted[row] = "".join([row_permuted[row][j] for j in back_perm])
            #                         permuted_rels.append(tuple(row_permuted))
            #                     new_rel.append(tuple(permuted_rels))

            # if len(new_obj) == 0:
            #     not_found = True

            # last_obj = new_obj
            # last_rel = new_rel

        if subsym_init.is_terminal():
            subsymbolic_success += 1
            os.system(f"rm -rf {img_out_path}/{n_action}/{i}")
        else:
            print(subsym_init.state[:, :3], file=open(os.path.join(img_out_path, str(n_action), str(i), "state.txt"), "w"))
            print(subsym_init.goal[:, :3], file=open(os.path.join(img_out_path, str(n_action), str(i), "state.txt"), "a"))

        # if sym_init.is_terminal():
        #     symbolic_success += 1

        # goal_obj_str = tuple(utils.binary_tensor_to_str(goal_obj[0]))
        # goal_rel_str = []
        # for rel in goal_rel[0]:
        #     goal_rel_str.append(tuple(utils.binary_tensor_to_str(rel)))
        # goal_rel_str = tuple(goal_rel_str)
        # for last_obj_i, last_rel_i in zip(last_obj, last_rel):
        #     if (last_obj_i == goal_obj_str) and (last_rel_i == goal_rel_str):
        #         dict_success += 1
        #         break

        # if not not_found:
        #     dict_found += 1

        print(f"n_action: {n_action}, i: {i+1}, subsym: {subsymbolic_success}, sym: {symbolic_success}, "
              f"dict: {dict_success}, dict found: {dict_found}", end="\r")
    print()
