import heapq
import torch
import tqdm

from explore_council import get_action_set
from dataset import StateActionEffectDataset
action_set = []
action_set_main,seperators,_ = get_action_set(8)
class Node:
    def __init__(self, state, cost, heuristic, parent, action):
        self.state = state
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent
        self.action = action
        self.depth = 0

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

target = None
def astar(start, goal, predict, heuristic, is_equal_func, max_depth = 3):
    open_set = []
    closed_set = set()

    start_node = Node(start, 0, heuristic(start, goal), None, None)
    heapq.heappush(open_set, start_node)

    while open_set:
        current_node = heapq.heappop(open_set)
        if is_equal_func(current_node.state, goal):
            path = []
            while current_node:
                path.append(current_node.action)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.state)

        if current_node.depth > max_depth:
            continue
        for action in action_set:
            neighbor_state = predict(current_node.state, action)
            # if neighbor_state in closed_set:
            #     continue

            cost = current_node.cost + 1  # Assuming a constant cost for each action
            heuristic_value = heuristic(neighbor_state, goal)
            new_node = Node(neighbor_state, cost, heuristic_value, current_node, action)
            new_node.parent = current_node
            new_node.depth = new_node.parent.depth + 1

            if new_node not in open_set and new_node.state not in closed_set:
                heapq.heappush(open_set, new_node)

    return None
def heuristic(state, goal):
    return ((state[:,:3] - goal[:,:3])**2).sum(dim = -1).sqrt().sum().item()

def is_equal(state1, state2, error=0.05):
    # Define a function to check if two states are equal
    return ((state1[:,:3] - state2[:,:3])**2).sum(dim = -1).sqrt().sum().item() < error* state1.shape[0]

model = None
def predict(state, action):
    global model

    sample = {}
    sample["state"] = state.unsqueeze(0)
    sample["action"] = action[:state.shape[0]].unsqueeze(0)
    sample["pad_mask"] = torch.ones((1,state.shape[0]))
    with torch.no_grad():
        _,_,e = model.forward(sample,eval_mode = True)
    e = e[0].cpu()
    obj1 = action[:,0].argmax()
    obj2 = action[:, 4].argmax()

    ds1 = action[obj1, [1,2]] * 0.075
    ds2 = action[obj2, [5,6]] * 0.075
    travel_distance = (state[obj2, :2] + ds1) - (state[obj1, :2]+ ds2)
    is_above = (e[:,2] > 0.3 ).int().unsqueeze(-1)
    next_state = state[:,:3].clone()
    next_state += e[:,:3] + e[:,7:10]
    next_state[:,:2] += travel_distance.repeat(state.shape[0],1) * is_above
    return next_state

def evaluate_dataset(model, dataset, max_depth): 
    counter = 0
    global action_set
    global target
    for i in range(len(dataset)):
        s = dataset[i]
        t = dataset[i + max_depth]
        mask = dataset.mask[i]
        state = s["state"]
        target = s["action"][:mask]
        action_set = action_set_main[:seperators[mask-1], :mask]
        post_state = t["state"]
        x = astar(state[:mask], post_state[ :mask], predict, heuristic, is_equal, max_depth)
        if x:
            counter += 1
    return counter / len(dataset)
if __name__ == "__main__":

     #TODO: set this according to dataset
    
    model = torch.load("./council/3_generation10.pt")
    # model = torch.load("./baseline.pt")
    model.eval_mode()
    accs = []
    for i in range(2):
        dataset = StateActionEffectDataset(f"long_horizon_collection_4", "test")
        acc = evaluate_dataset(model, dataset)
        print(acc)
        accs.append(acc)
    print(accs)