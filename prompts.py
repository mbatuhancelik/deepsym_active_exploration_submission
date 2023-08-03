from environment import BlocksWorld_v4
import numpy as np
import time
colors = ["red" , "green" , "blue", "purple" , "black" , "white" , "pink", "cyan", "brown", "gray",  "orange", "beige"]
type_mapping = {
    0: "sphere",
    1 : "cube",
    2 : "cylinder",
    3 : "tall prism",
    4 : "wide prism"
}
source_positions = {
    0: "center",
    1: "left",
    -1: "right"
}
target_positions = {
    0: "on top of",
    1: "next to",
    2: "in front of"
}
finishing_line = " \n Your output should be in this format:  <your reasoning why this action is selected> \n<selected sequence’s number>"
def prompt_object_list(types):
    obj = []
    for i, type in enumerate(types):
        obj.append(f"the {colors[i]} {type_mapping[type]}")
    return obj
def action_to_prompt(action, objects):
    source = objects[action[0]]
    target = objects[action[1]]
    source_position = source_positions[action[3]]
    target_position = target_positions[(action[5]) + (action[4]) * 2 ]
    if "wide" in target and action[5] != 0:
        target_position  = f"top {target_position}"
    return f"Put {source} {target_position} {target }"
    return f"[{source},{target},{target_position}]"    
    return f"Grasp {source} from {source_position} and {target_position } to {target}. "
def action_seq_prompt(seq, objects):
    prompt = ""
    for act in seq:
        prompt += action_to_prompt(act, objects) + "\n"
    return prompt
def generate_cases(cases):
    prompt = ""
    for i, c in enumerate(cases):
        prompt += f"{i + 1} ) {c}" 
    return prompt
def generate_description(objects):
    scene = "There is "
    for o in objects:
        if o == objects[-1]:
            scene += f"and a {o[4:]}"
        else:
            scene += f"a {o[4:]}, "
    scene += " in the current scene."
    return scene
def previous(prev, objects):
    p = "Previously executed actions:\n"
    for a in prev:
        p += action_to_prompt(a, objects) + "\n"
    return p + "\n"
def cases(env:BlocksWorld_v4, num_cases):
    seqs = []
    pairs = np.zeros((env.num_objects, env.num_objects))
    while len(seqs) < num_cases:
        act = env.sample_random_action()
        seqs.append([act])
        pairs[act[0], act[1]] += 1
    return seqs

def generate_prompt(env ,prev = None ,num_cases = 10, seq_len = 1):
    t = env.state_obj_poses_and_types()[1]
    objects = prompt_object_list(t)
    seqs = cases(env, num_cases)
    action_prompts = [action_seq_prompt(seq, objects) for seq in seqs]
    prompt = generate_description(objects) + "\n"
    prompt += prompt_contact_info(env, objects)
    if prev != None:
        if len(prev) > 0:
            prompt += previous(prev, objects)
    prompt += "Possible actions:\n"
    prompt += generate_cases(action_prompts)
    return prompt,seqs
def execute_sequence(env, seqs):
    for act in seq:
        env.step(*act, sleep = True)
def prompt_contact_info(env: BlocksWorld_v4, objects):   
    contact_graph, _  = env.update_contact_graph()
    prompt = ""
    for i in range(env.num_objects):
        for k in range(i + 1, env.num_objects ):
            pos = env.state_obj_poses()
            if contact_graph[i][k] == 1:
                z_i = pos[i][2]
                z_k = pos[k][2]
                if np.abs(z_i - z_k) < 0.01:
                    prompt += f"{objects[i]} is next to {objects[k]}.\n" 
                else:
                    if z_i > z_k:
                        prompt += f"{objects[i]} is stacked on {objects[k]}.\n"
                    else:
                         prompt += f"{objects[k]} is stacked on {objects[i]}.\n"
            elif np.abs(pos[i][2] - pos[k][2]) < 0.01:
                    if np.abs(pos[i][1]  - pos[k][1]) < (env.ds + 0.03 ) and np.abs(pos[i][0]  - pos[k][0]) < (0.01 ):
                        prompt += f"{objects[i]} is next to {objects[k]}.\n"
                    elif  np.abs(pos[i][0]  - pos[k][0]) < (env.ds + 0.03 ) and np.abs(pos[i][1]  - pos[k][1]) < (0.01 ) :
                        if pos[i][1]  > pos[k][1]:
                            prompt += f"{objects[i]} is in front of {objects[k]}.\n"
                        else :
                            prompt += f"{objects[k]} is in front of {objects[i]}.\n" 
    return prompt
#generates another prompt with action sequences shuffled
def generate_shuffled_prompt(seqs):
    np.random.shuffle(seqs)
    t = env.state_obj_poses_and_types()[1]
    objects = prompt_object_list(t)
    action_prompts = [action_seq_prompt(seq, objects) for seq in seqs]
    prompt = generate_description(objects) + "\n"
    prompt += generate_cases(action_prompts)
    return prompt
if __name__ == "__main__":
    env = BlocksWorld_v4(gui=1, min_objects=5, max_objects=5)
    while(True):
        prompt, seqs = generate_prompt(env, num_cases=10, seq_len=1)
        # with open("prompt", "w+") as f:
        #     f.write(
        #         prompt)
        print(prompt)
        #shuffled_prompt = generate_shuffled_prompt(seqs)
        #with open("prompt_shuffled", "w+") as f:
         #   f.write(
          #      "Remember, you are helping me to select the most curious action sequence."+
           #     shuffled_prompt+
            #    finishing_line)
        seq = seqs[int(input("Sequence number"))]
        execute_sequence(env, seq)
        