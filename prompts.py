from environment import BlocksWorld_v4
env = BlocksWorld_v4(gui=0, min_objects=3, max_objects=5)
colors = ["red" , "green" , "blue", "purple" , "orange" , "black" , "white" , "pink"]
type_mapping = {
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
    0: "top",
    1: "left",
    -1: "right"
}
def prompt_object_list(types):
    obj = []
    for i, type in enumerate(types):
        obj.append(f"{colors[i]} {type_mapping[type]}")
    return obj
def action_to_prompt(action, objects):
    source = objects[action[0]]
    target = objects[action[1]]
    source_position = source_positions[action[3]]
    target_position = target_positions[action[5]]
    if "wide" in target and action[5] != 0:
        target_position  = f"top {target_position}"
    return f"Grasp {source} from {source_position} and put it to {target_position } of {target}. "
    return f"Grasp: {source} , {source_position}. ut: {target } {target_position} "
def action_seq_prompt(seq, objects):
    prompt = ""
    for act in seq:
        prompt += action_to_prompt(act, objects) + "\n"
    return prompt
def generate_cases(cases):
    prompt = ""
    for i, c in enumerate(cases):
        prompt += f"{i}) {c}" + "\n"
    return prompt
def generate_description(objects):
    scene = "There is "
    for o in objects:
        scene += f"a {o}, "
    scene = scene[:-2]
    scene += " in the curret scene."
    return scene
def generate_prompt(num_cases = 3, seq_len = 2):
    env.reset_objects()
    t = env.state_obj_poses_and_types()[1]
    objects = prompt_object_list(t)
    seqs = [[env.sample_random_action() for i in range(seq_len)] for i in range(num_cases)]
    action_prompts = [action_seq_prompt(seq, objects) for seq in seqs]
    prompt = generate_description(objects) + "\n"
    prompt += generate_cases(action_prompts)
    return prompt,seqs
def execute_sequence(seq):
    for act in seq:
        env.step(*act)
if __name__ == "__main__":
    while(True):
        prompt, seqs = generate_prompt(num_cases=20, seq_len=5)
        print(prompt+"\nOutput only the selected sequence's number ")
        
        seq = seqs[int(input("Sequence number"))]
        execute_sequence(seq)
        