import time
import uuid
from copy import deepcopy

import numpy as np
import torch

import utils


class MCTSNode:
    def __init__(self, parent, state, forward_fn):
        self.name = uuid.uuid4().hex
        self.parent = parent
        self.state = state
        self.count = 0
        self.reward = 0
        self.is_terminal = state.is_terminal()
        self._forward_fn = forward_fn
        if self.is_terminal:
            self.actions = None
            self.children = None
        else:
            self.actions = state.get_available_actions()
            self.children = [None] * len(self.actions)

    def run(self, iter_limit, time_limit, default_depth_limit=10, default_batch_size=1):
        i = 0
        start = time.time()
        end = time.time()
        time_elapsed = end - start
        start_node_count, _ = self._tree_stats()
        while (i < iter_limit) and (time_elapsed < time_limit):
            v = self._tree_policy()
            reward = 0.0

            # sequential
            for _ in range(default_batch_size):
                reward += v._default_policy(default_depth_limit)
            reward /= default_batch_size

            v._backup(reward)

            i += 1
            end = time.time()
            time_elapsed = end - start
            if i % 100 == 0:
                node_count, depth = self._tree_stats()
                print(f"Tree depth={depth}, node count={node_count}, "
                      f"node/sec={(node_count-start_node_count)/time_elapsed:.2f}, "
                      f"best reward={self.reward/self.count}")

        return self.children_yield()

    def best_child_idx(self):
        idx = np.argmax(self.children_ucb1())
        return idx

    def best_child_for_plan(self):
        idx = np.argmax(self.children_yield())
        return idx

    def children_ucb1(self):
        if not self.is_terminal:
            scores = []
            for child in self.children:
                if child is not None:
                    probs = []
                    bounds = []
                    # there may be stochastic outcomes for the same action
                    for outcome in child:
                        probs.append(outcome.count)
                        bounds.append(outcome.UCB1())
                    probs = np.array(probs)
                    bounds = np.array(bounds)
                    probs = probs/probs.sum()
                    scores.append((probs * bounds).sum())
                else:
                    scores.append(np.inf)
            return scores
        else:
            return None

    def children_yield(self):
        if not self.is_terminal:
            scores = []
            for child in self.children:
                if child is not None:
                    probs = []
                    yields = []
                    # there may be stochastic outcomes for the same action
                    for outcome in child:
                        probs.append(outcome.count)
                        yields.append(outcome.reward/outcome.count)
                    probs = np.array(probs)
                    yields = np.array(yields)
                    probs = probs / probs.sum()
                    scores.append((probs * yields).sum())
                else:
                    scores.append(0)
            scores = np.array(scores)
            return scores
        else:
            return None

    def UCB1(self):
        if self.parent is None:
            return None
        else:
            N = self.parent.count
            score = self.reward/self.count + np.sqrt((2*np.log(N)) / self.count)
            return score

    def plan(self):
        if self.is_terminal:
            return self.state, "", [], 1.0
        idx = self.best_child_for_plan()
        if self.children[idx] is None:
            print("Plan not found.")
            return self.state, "", [(idx, 0)], 1.0
        elif len(self.children[idx]) == 1:
            child_state, child_plan_txt, child_plan, child_prob = self.children[idx][0].plan()
            return child_state, "_".join(filter(None, [self.actions[idx], child_plan_txt])), \
                [(idx, 0)]+child_plan, child_prob
        else:
            probs = []
            for out in self.children[idx]:
                probs.append(out.count)
            probs = np.array(probs)
            probs = probs / probs.sum()
            prob_max = np.argmax(probs)
            p = np.max(probs)
            child_state, child_plan_txt, child_plan, child_prob = self.children[idx][prob_max].plan()
            return child_state, "_".join(filter(None, [self.actions[idx], child_plan_txt])), \
                [(idx, prob_max)]+child_plan, p*child_prob

    def best_reward_path(self):
        if self.is_terminal:
            return self.state, "", [], 1.0
        idx = self.best_child_for_plan()
        if self.children[idx] is None:
            print("Plan not found.")
            return self.state, "", [(idx, 0)], 1.0
        elif len(self.children[idx]) == 1:
            child_state, child_plan_txt, child_plan, child_prob = self.children[idx][0].best_reward_path()
            return child_state, "_".join(filter(None, [self.actions[idx], child_plan_txt])), \
                [(idx, 0)]+child_plan, child_prob
        else:
            probs = []
            rewards = []
            for out in self.children[idx]:
                probs.append(out.count)
                rewards.append(out.reward)
            probs = np.array(probs)
            probs = probs / probs.sum()
            rewards = np.array(rewards)
            best_idx = np.argmax(rewards)
            p = probs[best_idx]
            child_state, child_plan_txt, child_plan, child_prob = self.children[idx][best_idx].best_reward_path()
            return child_state, "_".join(filter(None, [self.actions[idx], child_plan_txt])), \
                [(idx, best_idx)]+child_plan, p*child_prob

    def _expand(self):
        idx = self.children.index(None)
        action = self.actions[idx]
        # ACT HERE #
        next_state = self._forward_fn.forward(self.state, action)
        self.children[idx] = [MCTSNode(parent=self,
                                       state=next_state,
                                       forward_fn=self._forward_fn)]
        ############
        return self.children[idx][0]

    def _tree_policy(self):
        # if there is an unexpanded node, first expand it.
        if self.is_terminal:
            return self
        if None in self.children:
            return self._expand()
        # else choose the best child by UCB1
        else:
            # have to change here
            idx = self.best_child_idx()
            next_state = self._forward_fn.forward(self.state, self.actions[idx])
            children_states = list(map(lambda x: x.state, self.children[idx]))
            result, out_idx = utils.in_array(next_state, children_states)
            if not result:
                self.children[idx].append(MCTSNode(parent=self,
                                                   state=next_state,
                                                   forward_fn=self._forward_fn))
                return self.children[idx][-1]._tree_policy()
            else:
                return self.children[idx][out_idx]._tree_policy()

    def _default_policy(self, depth_limit):
        if (not self.is_terminal) and (depth_limit > 0):
            random_action = np.random.choice(self.actions)
            # ACT HERE #
            next_state = self._forward_fn.forward(self.state, random_action)
            v = MCTSNode(parent=None, state=next_state, forward_fn=self._forward_fn)
            ############
            return v._default_policy(depth_limit-1)
        else:
            return self.state.reward()

    def _backup(self, reward):
        if self.parent is not None:
            self.count += 1
            self.reward += reward
            self.parent._backup(reward)
        else:
            self.count += 1
            self.reward += reward

    def __repr__(self):
        state = str(self.state)
        children = []
        if not self.is_terminal:
            children_scores = list(map(lambda x: "%.2f" % x, self.children_ucb1()))
            for c in self.children:
                if c is None:
                    children.append("None")
                else:
                    outcomes = list(map(lambda x: x.name, c))
                    outcomes = "[" + ", ".join(outcomes) + "]"
                    children.append(outcomes)
        string = "Name: " + self.name + "\n"
        if self.parent:
            string += "Parent: " + self.parent.name + "\n"
        else:
            string += "Parent: None\n"
        if not self.is_terminal:
            string += "Children: [" + ", ".join(children) + "]\n"
            string += "Children UCB1: [" + ", ".join(children_scores) + "]\n"
        string += "State:\n" + state + "\n"
        string += "Reward: " + str(self.reward) + "\n"
        string += "Count: " + str(self.count) + "\n"
        string += "Terminal: " + str(self.is_terminal)
        return string

    def _tree_stats(self):
        if self.is_terminal:
            return 1, 0

        children_depths = []
        total_nodes = 1
        for c in self.children:
            if c is None:
                children_depths.append(0)
            elif len(c) == 1:
                nodes, depth = c[0]._tree_stats()
                children_depths.append(depth)
                total_nodes += nodes
            else:
                gchild_depths = []
                for c_i in c:
                    nodes, depth = c_i._tree_stats()
                    gchild_depths.append(depth)
                    total_nodes += nodes
                children_depths.append(max(gchild_depths)+1)
        return total_nodes, max(children_depths)+1


class MCTSState:
    def __init__(self, goal):
        pass

    def reward(self):
        pass

    def get_available_actions(self):
        pass

    def is_terminal(self):
        pass

    def __repr__(self):
        pass

    def is_equal(self, other):
        pass


class MCTSForward:
    def __init__(self):
        pass

    def forward(self, state, action):
        pass


class SymbolicForwardModel(MCTSForward):
    def __init__(self, model):
        self.model = model

    def forward(self, state, action):
        obj_symbol, rel_symbol = utils.to_tensor_state(state.state)
        action = torch.tensor([int(a_i) for a_i in action.split(",")])
        action_placeholder = torch.zeros(obj_symbol.shape[0], 8)  # (grasp_or_release, dx_loc, dy_loc, rot)
        action_placeholder[action[0], :4] = torch.tensor([1, action[1], action[2], 1], dtype=torch.float)
        action_placeholder[action[3], 4:] = torch.tensor([1, action[4], action[5], 1], dtype=torch.float)
        z_cat = torch.cat([obj_symbol, action_placeholder], dim=-1)
        with torch.no_grad():
            obj_symbol_next, rel_symbol_next = self.model(z_cat.unsqueeze(0), rel_symbol.unsqueeze(0))
        obj_symbol_next = obj_symbol_next.sigmoid().bernoulli()[0]
        rel_symbol_next = rel_symbol_next.sigmoid().bernoulli()[0]
        str_state = utils.to_str_state(obj_symbol_next, rel_symbol_next, torch.ones(obj_symbol_next.shape[0]))
        return SymbolicState(state=deepcopy(str_state), goal=deepcopy(state.goal))


class SymbolicState(MCTSState):
    def __init__(self, state, goal):
        self.state = state
        self.goal = goal

    def reward(self):
        reward = int(self.is_terminal())
        return reward

    def get_available_actions(self):
        o_str, _ = self.state.split("_")
        o_arr = o_str.split("-")
        n_obj = len(o_arr)
        actions = []
        for i in range(n_obj):
            for iy in range(-1, 2):
                for j in range(n_obj):
                    for jy in range(-1, 2):
                        actions.append(f"{i},0,{iy},{j},0,{jy}")
        return actions

    def is_terminal(self):
        return self.state == self.goal

    def is_equal(self, other):
        return self.state == other.state

    def __repr__(self):
        return self.state


class SubsymbolicState(MCTSState):
    threshold = 0.03

    def __init__(self, state, goal):
        self.state = state
        self.goal = goal

    def reward(self):
        return -self.goal_diff()

    def goal_diff(self):
        diff = (self.state[:, :3] - self.goal[:, :3]).abs().sum()
        return diff

    def get_available_actions(self):
        n_obj = self.state.shape[0]
        actions = []
        for i in range(n_obj):
            for iy in range(-1, 2):
                for j in range(n_obj):
                    for jy in range(-1, 2):
                        actions.append(f"{i},0,{iy},{j},0,{jy}")
        return actions

    def is_terminal(self):
        diff = self.goal_diff()
        return diff < SubsymbolicState.threshold

    def is_equal(self, other):
        diff = (self.state[:, :3] - other.state[:, :3]).abs().sum()
        return diff < SubsymbolicState.threshold

    def __repr__(self):
        return self.state.__repr__()


class SubsymbolicForwardModel(MCTSForward):
    def __init__(self, model):
        self.model = model

    def forward(self, state, action):
        n_objs = state.state.shape[0]
        mask = torch.ones(1, n_objs)
        action = torch.tensor([int(a_i) for a_i in action.split(",")])
        action_placeholder = torch.zeros(state.state.shape[0], 8)  # (grasp_or_release, dx_loc, dy_loc, rot)
        action_placeholder[action[0], :4] = torch.tensor([1, action[1], action[2], 1], dtype=torch.float)
        action_placeholder[action[3], 4:] = torch.tensor([1, action[4], action[5], 1], dtype=torch.float)
        inp = {
            "state": state.state.unsqueeze(0),
            "action": action_placeholder.unsqueeze(0),
            "pad_mask": mask
        }
        with torch.no_grad():
            _, _, e = self.model.forward(inp, eval_mode=False)
            delta_pos = state.state[action[3]] - state.state[action[0]]
            dx, dy = delta_pos[0], delta_pos[1]
            dx += (-action[1] * 0.075) + (action[4] * 0.075)
            dy += (-action[2] * 0.075) + (action[5] * 0.075)
            next_state = state.state.clone()
        next_state[:, :3] = next_state[:, :3] + e[0, :, :3] + e[0, :, 3:]
        # print(dx, dy, e[0, :, :3], e[0, :, 3:])
        for i in range(n_objs):
            # if the object is lifted, it should be moved
            if e[0, i, 2] > 0.1:
                next_state[i, 0] += dx
                next_state[i, 1] += dy

        return SubsymbolicState(state=next_state, goal=state.goal)
