import argparse
import os

import gym
import crafter
import torch


inventory = {
    "health": 21,
    "food": 22,
    "drink": 23,
    "energy": 24,
    "sapling": 25,
    "wood": 26,
    "stone": 27,
    "coal": 28,
    "iron": 29,
    "diamond": 30,
    "wood_pickaxe": 31,
    "stone_pickaxe": 32,
    "iron_pickaxe": 33,
    "wood_sword": 34,
    "stone_sword": 35,
    "iron_sword": 36
}


def make_state(info):
    a_x, a_y = info["player_pos"]
    x = torch.tensor(info["semantic"][a_x-3:a_x+4, a_y-4:a_y+5], dtype=torch.uint8)
    x = x.reshape(-1)
    x = eye2[x.tolist()].reshape(7, 9, eye2.shape[0])
    x_idx = torch.arange(7).repeat_interleave(9, 0).reshape(7, 9, 1)
    y_idx = torch.arange(9).repeat(7, 1).unsqueeze(2)
    x = torch.cat([x, x_idx, y_idx, torch.zeros(7, 9, 16)], dim=-1)
    for key in info["inventory"]:
        idx = inventory[key]
        value = info["inventory"][key]
        x[:, :, idx] = value
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore the Crafter environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # might have saved them into the folder, and also represent action
    # directly as an integer, not a vector. but whatever.
    state = torch.zeros(args.N, 7, 9, 37, dtype=torch.uint8)
    action = torch.zeros(args.N, 17, dtype=torch.uint8)
    effect = torch.zeros(args.N, 7, 9, 37, dtype=torch.int16)
    eye = torch.eye(17, dtype=torch.uint8)
    eye2 = torch.eye(19, dtype=torch.uint8)

    env = gym.make("CrafterReward-v1")
    it = 0
    while it < args.N:
        env.reset()
        a = env.action_space.sample()
        _, _, done, info = env.step(a)
        prev_state = make_state(info)
        while not done:
            a = env.action_space.sample()
            _, _, done, info = env.step(a)
            new_state = make_state(info)
            e = new_state - prev_state

            state[it] = prev_state.clone()
            action[it] = eye[a]
            effect[it] = e
            prev_state = new_state.clone()

            it += 1
            if it == args.N:
                break

            if it % (args.N // 20) == 0:
                print(f"{100*it//args.N}% completed")

    torch.save(state, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(action, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(effect, os.path.join(args.o, f"effect_{args.i}.pt"))
