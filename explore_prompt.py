import time
import os
import argparse

import torch
import numpy as np
import environment
import subprocess
import openai
from prompts import generate_prompt

buffer_type = ""
buffer_lenght = 0
openai.api_key='insert key here'
failed_tries = 0
with open("system_prompt.md") as f:
    system_prompt = f.read()

states = None
actions = None
masks = None
effects = None
post_states = None
args = None
i = 0
unsuccessfull_attempts = 0
messages = [
                    {"role": "system", "content": system_prompt}, 
                ]
action_seq = []
def printLog(s, file = ""):
    with open(os.path.join(args.o, f"logs{args.i}_{file}.out"), "a+") as f:
        f.write(str(s) +'\n')
def clear_zsh_output():
    # The escape code for clearing the terminal screen in zsh
    # \033 is the escape character, [2J is the code for clearing the screen, and H sets the cursor to the top-left corner
    subprocess.run(['clear'])
def message_list_to_text():
    t = "===============\n"
    for m in messages:
        if m["role"] == "system":
            continue
        t += f"{m['role']} : {m['content']}\n"
    return t
def reset_messages():
    global messages
    global system_prompt
    global args
    with open(os.path.join(args.o, f"dialogue.out"), "a+") as f:
        f.write(message_list_to_text())
    messages = [
                    {"role": "system", "content": system_prompt + "\n Note: Do not put colors into consideration when making your choice."}, 
                ]
def get_action(env:environment.BlocksWorld_v4):
    global states
    global actions
    global masks
    global effects
    global post_states
    global args
    global unsuccessfull_attempts
    global messages
    global action_seq
    global i
    action = None
    while action is None:
        reset_messages()
        prompt, seqs = generate_prompt(env,prev=action_seq ,seq_len=1, num_cases=10)
        messages.append({"role" : "user", "content": prompt})
        response = "None"
        try:
            l=0
            for m in messages:
               l += len(m["content"])
            printLog(f"message len: {l}", file = "sizes") 
            print(f"[User]: {m['content']}\n")
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                temperature = 0,
                messages=messages,
                max_tokens = 1500
                    )
            response = chat_completion.choices[0].message.content
            print(f"[Assistant]: {response}\n")
            # response = "Selected action is: a) \n"
            for sentence in response.split("\n"):
                sentence = sentence.lower()
                if "selected action is" in sentence:
                    post_tokens = sentence.split("selected action is")
                    index = None
                    for c in post_tokens[-1]:
                        if c == ':': 
                            continue
                        k = ord(c) - ord('1')
                        if k >= 0 and k < len(seqs):
                            index = k
                            break
                    action = seqs[index]
            messages.append({"role" : "assistant", "content" :response})
            printLog(len(system_prompt) + len(prompt) + len(response), file = "sizes")
            action_seq.append(action[0])
            unsuccessfull_attempts = 0
            return action[0]
        except Exception as e:
            errm = str(e)
            printLog("===========")
            printLog("error:")
            printLog(errm)
            printLog("response:")
            printLog(response)
            printLog("===========")
            if "Rate limit reached" in errm:
                exit(1)
            unsuccessfull_attempts += 1
            if states is not None:
                torch.save(states[:i], os.path.join(args.o, f"state_{args.i}.pt"))
                torch.save(actions[:i], os.path.join(args.o, f"action_{args.i}.pt"))
                torch.save(masks[:i], os.path.join(args.o, f"mask_{args.i}.pt"))
                torch.save(effects[:i], os.path.join(args.o, f"effect_{args.i}.pt"))
                torch.save(post_states[:i], os.path.join(args.o, f"post_state_{args.i}.pt"))
                torch.save(contact[:i], os.path.join(args.o, f"contact_{args.i}.pt"))
                torch.save(clusters[:i], os.path.join(args.o, f"clusters_{args.i}.pt"))
            if unsuccessfull_attempts == 20:
                exit(2)
def collect_rollout(env):
    action = get_action(env)
    
    position, effect, types = env.step(*action, sleep = True)
    post_position, _ = env.state_obj_poses_and_types()
    return position, types, action, effect, post_position


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-T", help="interaction per episode", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    env = environment.BlocksWorld_v4(gui=1, min_objects=4, max_objects=4)
    np.random.seed()
    if args.i == 0:
        with  open(os.path.join(args.o, f"system_prompt.out"), "w+") as f:
            f.write(system_prompt)
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
    contact = torch.zeros(args.N, env.max_objects, env.max_objects, dtype=torch.float)
    clusters = torch.zeros(args.N, env.max_objects, dtype=torch.float)

    prog_it = args.N
    start = time.time()
    env_it = 0
    i = 0
    reset_messages()
    input()
    while i < args.N:
        clear_zsh_output()
        print(f"[System]: {system_prompt}\n") 
        env_it += 1
        cont , cl = env.update_contact_graph()
        contact[i, :env.num_objects, :env.num_objects] = torch.tensor(cont)
        clusters[i, :env.num_objects] = torch.tensor(cl)
        
        position_pre, obj_types, action, effect, position_post = collect_rollout(env)
        
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
            reset_messages()
            action_seq = []

        i += 1
        if i % prog_it == 0:
            printLog(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(masks, os.path.join(args.o, f"mask_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    torch.save(post_states, os.path.join(args.o, f"post_state_{args.i}.pt"))
    torch.save(contact, os.path.join(args.o, f"contact_{args.i}.pt"))
    torch.save(clusters, os.path.join(args.o, f"clusters_{args.i}.pt"))
    end = time.time()
    del env
    printLog(f"Completed in {end-start:.2f} seconds.")
