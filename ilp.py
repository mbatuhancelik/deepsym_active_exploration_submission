import torch
from swiplserver import PrologMQI

from mcts import MCTSNode


obj_pre = torch.load("save/only_xyz/train_z_obj_pre.pt")
rel_pre = torch.load("save/only_xyz/train_z_rel_pre.pt")
action = torch.load("save/only_xyz/train_z_act.pt")
obj_post = torch.load("save/only_xyz/train_z_obj_post.pt")
rel_post = torch.load("save/only_xyz/train_z_rel_post.pt")
mask = torch.load("save/only_xyz/train_mask.pt")


def select_from_obj(name, obj_value, cond):
    return f"nth0({name}, Obj{cond}, Obj{name}{cond}), Obj{name}{cond} = {list(obj_value)}"


def select_from_action(name, action_value):
    return f"nth0({name}, Action, Act{name}), Act{name} = {list(action_value)}"


def select_from_rel(rel_idx, name, target_name, rel_value, cond):
    if target_name == "all":
        return f"nth0({name}, Rel{cond}, Rel{name}{cond}), \
forall(member(Rel{name}{target_name}{cond}, Rel{name}{cond}), \
nth0({rel_idx}, Rel{name}{target_name}{cond}, Rel{rel_idx}{name}{target_name}{cond}) \
Rel{rel_idx}{name}{target_name}{cond} = {rel_value})"
    else:
        return f"nth0({name}, Rel{cond}, Rel{name}{cond}), \
nth0({target_name}, Rel{name}{cond}, Rel{name}{target_name}{cond}), \
nth0({rel_idx}, Rel{name}{target_name}{cond}, Rel{rel_idx}{name}{target_name}{cond}), \
Rel{rel_idx}{name}{target_name}{cond} = {rel_value}"


class Rule:
    obj_values = [tuple(v) for v in obj_pre.flatten(0, 1).int().unique(dim=0).tolist()]
    action_values = [tuple(v) for v in action.flatten(0, 1).int().unique(dim=0).tolist()]
    relation_values = [0, 1]
    num_relations = 4
    max_args = 3

    def __init__(self, variables, obj_pre_bound, obj_post_bound, rel_post_bound, applied_actions):
        self.variables = variables
        self.obj_pre_bound = obj_pre_bound
        self.obj_post_bound = obj_post_bound
        self.rel_post_bound = rel_post_bound
        self.applied_actions = applied_actions

    def get_available_actions(self):
        if len(self.applied_actions) > 0 and self.applied_actions[-1][0] == "end":
            return []

        actions = [("end",)]
        n_vars = len(self.variables)
        var_name = f"A{n_vars}"

        # qualifiers:
        if n_vars < Rule.max_args:
            # select an object with a specific symbol
            for val in Rule.obj_values:
                actions.append(("select_from_obj", var_name, val, "Pre"))

            # select an action with a specific symbol
            for val in Rule.action_values:
                actions.append(("select_from_action", var_name, val))

            # select an object that has a specific relation with all other objects
            # for val in self.relation_values:
            #     for i in range(self.num_relations):
            #         actions.append(("select_from_rel", i, var_name, "all", val, "Pre"))

            for val in self.obj_pre_bound.values():
                # select an object that has the same symbol as another object
                actions.append(("select_from_obj", var_name, val, "Pre"))

            # select an object that has a specific relation with another object
            for var in self.variables:
                for val in Rule.relation_values:
                    for i in range(Rule.num_relations):
                        actions.append(("select_from_rel", i, var_name, var, val, "Pre"))

        # constraints:
        # assert the post-symbol of an object to a value
        for var in self.variables:
            for val in Rule.obj_values:
                actions.append(("select_from_obj", var, val, "Post"))

        # assert the post-symbol of an object to the pre-symbol of another object
        for var in self.variables:
            for val in self.obj_pre_bound.values():
                actions.append(("select_from_obj", var, val, "Post"))

        # assert the post-symbol of an object to the post-symbol of another object
        for var in self.variables:
            for val in self.obj_post_bound.values():
                actions.append(("select_from_obj", var, val, "Post"))

        # assert the post-rel of an object-object pair to a value
        for var1 in self.variables:
            for var2 in self.variables:
                for val in Rule.relation_values:
                    for i in range(Rule.num_relations):
                        actions.append(("select_from_rel", i, var1, var2, val, "Post"))

        # assert the post-rel of an object-all pair to a value
        # for var1 in self.variables:
        #     for val in self.relation_values:
        #         for i in range(self.num_relations):
        #             actions.append(("select_from_rel", i, var1, "all", val, "Post"))

        return actions

    def is_terminal(self):
        if len(self.applied_actions) > 0 and self.applied_actions[-1][0] == "end":
            return True
        else:
            return False

    def is_equal(self, other):
        if len(self.applied_actions) != len(other.applied_actions):
            return False
        for a1, a2 in zip(self.applied_actions, other.applied_actions):
            if a1 != a2:
                return False
        return True

    def convert_to_query(self):
        query = ["experience(ObjPre, RelPre, Action, ObjPost, RelPost)"]
        actions = self.applied_actions.copy()
        if len(actions) > 0 and actions[-1][0] != "end":
            actions.append(("end",))
        elif len(actions) == 0:
            actions.append(("end",))

        for action in actions:
            if action[0] == "end":
                query.append(f"all_different([{', '.join(self.variables)}])")
                query.append(f"ObjDict = [{', '.join([f'{k}-{v}' for k, v in self.obj_post_bound.items()])}]")
                query.append(f"RelDict = [{', '.join([f'{r}-{q}-{k}-{v}' for (r, q, k), v in self.rel_post_bound.items()])}]")
                query.append("bind_rest(ObjDict, RelDict, ObjPre, RelPre, ObjPost, RelPost)")
            else:
                query.append(eval(action[0])(*action[1:]))
        query = ",".join(query)
        return query

    def reward(self):
        query = self.convert_to_query()
        with PrologMQI() as mqi:
            with mqi.create_thread() as thread:
                thread.query("consult('kb_res.pro')")
                thread.query("consult('utils.pro')")
                result = thread.query(query)
        if not result:
            reward = 0
        else:
            reward = len(result) / 13925
            print(query)
            print(len(result))
        del result
        return reward


class RuleForward:
    def __init__(self):
        pass

    def __call__(self, rule, action):
        if action[0] == "end":
            applied_actions = rule.applied_actions.copy() + [action]
            new_rule = Rule(
                variables=rule.variables.copy(),
                obj_pre_bound=rule.obj_pre_bound.copy(),
                obj_post_bound=rule.obj_post_bound.copy(),
                rel_post_bound=rule.rel_post_bound.copy(),
                applied_actions=applied_actions
            )
        else:
            applied_actions = rule.applied_actions.copy() + [action]
            new_rule = Rule(
                variables=rule.variables.copy(),
                obj_pre_bound=rule.obj_pre_bound.copy(),
                obj_post_bound=rule.obj_post_bound.copy(),
                rel_post_bound=rule.rel_post_bound.copy(),
                applied_actions=applied_actions
            )
            if action[0] == "select_from_obj":
                if action[1] not in new_rule.variables:
                    new_rule.variables.append(action[1])
                if action[3] == "Pre":
                    new_rule.obj_pre_bound[action[1]] = action[2]
                elif action[3] == "Post":
                    new_rule.obj_post_bound[action[1]] = action[2]
            elif action[0] == "select_from_rel":
                if action[2] not in new_rule.variables:
                    new_rule.variables.append(action[2])
                if action[3] not in new_rule.variables:
                    new_rule.variables.append(action[3])
                if action[5] == "Post":
                    new_rule.rel_post_bound[(action[1], action[2], action[3])] = action[4]
            elif action[0] == "select_from_action":
                if action[1] not in new_rule.variables:
                    new_rule.variables.append(action[1])
        return new_rule


if __name__ == "__main__":
    rule_forward = RuleForward()
    state = Rule(variables=[],
                 obj_pre_bound={},
                 obj_post_bound={},
                 rel_post_bound={},
                 applied_actions=[])

    root = MCTSNode(node_id=0, parent=None, state=state, forward_fn=rule_forward)
    root.run(iter_limit=10000, time_limit=3600, default_depth_limit=1, default_batch_size=1, n_proc=8)
