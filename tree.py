from collections import namedtuple


import torch
import numpy as np


Node = namedtuple("Node", ["left", "right", "object_bindings", "action_bindings", "relation_bindings",
                           "counts", "dataset_mask"])


def create_effect_classes(loader):
    # changed_vars = []
    effect_to_class = {}
    class_idx = 0
    effects = []
    for i, (obj_pre, rel_pre, _, obj_post, rel_post, mask) in enumerate(loader):
        obj_pre = obj_pre[0, mask[0]]
        obj_post = obj_post[0, mask[0]]
        obj_diff_idx = torch.where(obj_pre != obj_post)[0]
        obj_diff_idx = torch.unique(obj_diff_idx)
        obj_effects = []
        var_list = []
        for idx in obj_diff_idx:
            obj_effects.append(tuple(obj_post[idx].int().tolist()))
            var_list.append(idx.item())
        obj_effects = sorted(obj_effects)

        mm = (mask.T.float() @ mask.float()).bool()
        c = mask.sum()
        rel_pre = rel_pre[0, :, mm].reshape(4, c, c)
        rel_post = rel_post[0, :, mm].reshape(4, c, c)
        rel_diffs = torch.where(rel_pre != rel_post)
        rel_effects = [[], [], [], []]  # for four relations
        for rel, obj1, obj2 in zip(*rel_diffs):
            rel_value = rel_post[rel, obj1, obj2]
            tup = (rel_value.int().item(),)
            if tup not in rel_effects[rel.item()]:
                rel_effects[rel.item()].append(tup)
            if obj1.item() not in var_list:
                var_list.append(obj1.item())
            if obj2.item() not in var_list:
                var_list.append(obj2.item())
        rel_effects = [sorted(x) for x in rel_effects]
        # changed_vars.append(var_list)

        key = (tuple(obj_effects),) + tuple(tuple(x) for x in rel_effects)
        if key not in effect_to_class:
            effect_to_class[key] = class_idx
            class_idx += 1
        effects.append(effect_to_class[key])

    return effects, effect_to_class


def get_top_classes(sorted_effect_counts, perc, dataset_size):
    total_count = 0
    selected_keys = []
    for key in sorted_effect_counts:
        count = sorted_effect_counts[key]
        total_count += count
        selected_keys.append(key)
        if total_count/dataset_size >= perc:
            break
    return selected_keys


def get_effect_counts(effects, mask):
    effect_counts = {}
    for i, e in enumerate(effects):
        if not mask[i]:
            if e not in effect_counts:
                effect_counts[e] = 0
            effect_counts[e] += 1
    effect_counts = dict(sorted(effect_counts.items(), key=lambda x: x[1], reverse=True))
    return effect_counts


def filter_effect_classes(effects, selected_classes):
    filtered_effects = []
    for e in effects:
        if e in selected_classes:
            filtered_effects.append(e)
        else:
            filtered_effects.append(-1)
    return filtered_effects


def matrix_to_tuple(matrix):
    return tuple([tuple(row) for row in matrix])


def preprocess_data(o_i, r_i, a, o_f, r_f, m):
    o_i = o_i[0, m[0]].int()
    o_f = o_f[0, m[0]].int()
    c = m.sum()
    mm = (m.T.float() @ m.float()).bool()
    r_i = r_i[0, :, mm].reshape(4, c, c).int()
    r_f = r_f[0, :, mm].reshape(4, c, c).int()
    a = a[0, m[0]].int()
    return o_i, r_i, a, o_f, r_f


def check_rule(object_bindings, action_bindings, relation_bindings, loader, effects, dataset_mask):
    class_counts = {}
    classified_mask = np.zeros(len(dataset_mask), dtype=bool)
    for i, (o_i, r_i, a, o_f, r_f, m) in enumerate(loader):
        if not dataset_mask[i]:
            o_i, r_i, a, _, _ = preprocess_data(o_i, r_i, a, o_f, r_f, m)

            # get possible object indices
            obj_exists = True
            obj_possible_indices = {}
            for name in object_bindings:
                indices = torch.where((o_i == object_bindings[name]).all(dim=1))[0]
                if len(indices) > 0:
                    obj_possible_indices[name] = indices
                else:
                    obj_exists = False
                    break

            # get possible action indices
            act_exists = True
            act_possible_indices = {}
            for name in action_bindings:
                indices = torch.where((a == action_bindings[name]).all(dim=1))[0]
                if len(indices) > 0:
                    act_possible_indices[name] = indices
                else:
                    act_exists = False
                    break

            # constraints
            obj_act_binded = True
            all_names = list(set(list(object_bindings.keys()) + list(action_bindings.keys())))
            filtered_possible_indices = {}
            for name in all_names:
                if name in obj_possible_indices:
                    obj_indices = obj_possible_indices[name]
                else:
                    obj_indices = None

                if name in act_possible_indices:
                    act_indices = act_possible_indices[name]
                else:
                    act_indices = None

                if obj_indices is None and act_indices is None:
                    obj_act_binded = False
                    break
                elif obj_indices is None:
                    filtered_possible_indices[name] = act_indices
                elif act_indices is None:
                    filtered_possible_indices[name] = obj_indices
                else:
                    filtered_possible_indices[name] = np.intersect1d(obj_indices.numpy(), act_indices.numpy())

                if len(filtered_possible_indices[name]) == 0:
                    obj_act_binded = False
                    break

            possible_bindings = []
            if obj_act_binded:
                tensors = []
                for name in all_names:
                    tensors.append(filtered_possible_indices[name])
                bindings = torch.cartesian_prod(*tensors)
                if bindings.ndim == 1:
                    bindings = bindings.unsqueeze(1)
                num_vars = len(all_names)
                for binding in bindings:
                    if torch.unique(binding).shape[0] == num_vars:
                        possible_bindings.append({all_names[i]: binding[i] for i in range(num_vars)})
                if len(possible_bindings) == 0:
                    obj_act_binded = False

            rel_filtered_bindings = []
            for binding in possible_bindings:
                binding_valid = True
                for (rel_idx, name1, name2) in relation_bindings:
                    val = relation_bindings[(rel_idx, name1, name2)]
                    name1_idx = binding[name1]
                    name2_idx = binding[name2]
                    if r_i[rel_idx, name1_idx, name2_idx] != val:
                        binding_valid = False
                        break
                if binding_valid:
                    rel_filtered_bindings.append(binding)
            rel_exists = len(rel_filtered_bindings) > 0
            # TODO: check bindings in rel_filtered_bindings with the effect indices

            if obj_exists and act_exists and obj_act_binded and rel_exists:
                if effects[i] not in class_counts:
                    class_counts[effects[i]] = 0
                class_counts[effects[i]] += 1
                classified_mask[i] = True
        else:
            classified_mask[i] = True
    return class_counts, classified_mask


def calculate_entropy(counts):
    probs = {k: counts[k]/np.sum(list(counts.values())) for k in counts}
    entropy = -np.sum([probs[k]*np.log(probs[k]) for k in probs])
    return entropy


def expand_best_node(node, loader, effects, unique_object_values, unique_action_values, min_samples_split):
    """
    Expands the best node by binding variables to new values and creating new nodes.

    Args:
        node (Node): The node to expand.
        loader (torch.utils.data.DataLoader): The data loader.
        effects (List[int]): The effects.
        min_samples_split (int): The minimum number of samples required to split a node.

    Returns:
        Tuple[float, Node]: The entropy and the best node.
    """
    best_node = None
    best_entropy = 1e10

    obj_var_list = list(node.object_bindings.keys())
    act_var_list = list(node.action_bindings.keys())

    # if a new object variable is needed
    max_obj_idx = max([int(obj_var[3:]) for obj_var in obj_var_list]) if len(obj_var_list) > 0 else -1
    max_act_idx = max([int(act_var[3:]) for act_var in act_var_list]) if len(act_var_list) > 0 else -1
    new_obj_idx = max(max_obj_idx, max_act_idx) + 1
    new_obj_name = "obj" + str(new_obj_idx)

    # bind a variable in action list to a new object value
    for act_var in act_var_list:
        # continue if the variable already is bound to an object value
        if act_var in node.object_bindings:
            continue

        # bind the variable to each object value
        for obj_val in unique_object_values:
            object_bindings = node.object_bindings.copy()
            object_bindings[act_var] = obj_val
            counts, classified = check_rule(object_bindings=object_bindings,
                                            action_bindings=node.action_bindings,
                                            relation_bindings=node.relation_bindings,
                                            loader=loader,
                                            effects=effects,
                                            dataset_mask=node.dataset_mask)
            entropy = calculate_entropy(counts)
            if entropy < best_entropy and np.sum(classified) >= min_samples_split:
                best_node = Node(left=None, right=None,
                                 object_bindings=object_bindings,
                                 action_bindings=node.action_bindings.copy(),
                                 relation_bindings=node.relation_bindings.copy(),
                                 counts=counts,
                                 dataset_mask=classified)
                best_entropy = entropy

    # bind a new variable to a new object value
    for obj_val in unique_object_values:
        object_bindings = node.object_bindings.copy()
        object_bindings[new_obj_name] = obj_val
        counts, classified = check_rule(object_bindings=object_bindings,
                                        action_bindings=node.action_bindings,
                                        relation_bindings=node.relation_bindings,
                                        loader=loader,
                                        effects=effects,
                                        dataset_mask=node.dataset_mask)
        entropy = calculate_entropy(counts)
        if entropy < best_entropy and np.sum(classified) >= min_samples_split:
            best_node = Node(left=None, right=None,
                             object_bindings=object_bindings,
                             action_bindings=node.action_bindings.copy(),
                             relation_bindings=node.relation_bindings.copy(),
                             counts=counts,
                             dataset_mask=classified)
            best_entropy = entropy

    # bind a variable in object list to a new action value
    for obj_var in obj_var_list:
        # continue if the variable already is bound to an action value
        if obj_var in node.action_bindings:
            continue

        # bind the variable to each action value
        for act_val in unique_action_values:
            action_bindings = node.action_bindings.copy()
            action_bindings[obj_var] = act_val
            counts, classified = check_rule(object_bindings=node.object_bindings,
                                            action_bindings=action_bindings,
                                            relation_bindings=node.relation_bindings,
                                            loader=loader,
                                            effects=effects,
                                            dataset_mask=node.dataset_mask)
            entropy = calculate_entropy(counts)
            if entropy < best_entropy and np.sum(classified) >= min_samples_split:
                best_node = Node(left=None, right=None,
                                 object_bindings=node.object_bindings.copy(),
                                 action_bindings=action_bindings,
                                 relation_bindings=node.relation_bindings.copy(),
                                 counts=counts,
                                 dataset_mask=classified)
                best_entropy = entropy

    # bind a new variable to a new action value
    for act_val in unique_action_values:
        action_bindings = node.action_bindings.copy()
        action_bindings[new_obj_name] = act_val
        counts, classified = check_rule(object_bindings=node.object_bindings,
                                        action_bindings=action_bindings,
                                        relation_bindings=node.relation_bindings,
                                        loader=loader,
                                        effects=effects,
                                        dataset_mask=node.dataset_mask)
        entropy = calculate_entropy(counts)
        if entropy < best_entropy and np.sum(classified) >= min_samples_split:
            best_node = Node(left=None, right=None,
                             object_bindings=node.object_bindings.copy(),
                             action_bindings=action_bindings,
                             relation_bindings=node.relation_bindings.copy(),
                             counts=counts,
                             dataset_mask=classified)
            best_entropy = entropy

    # bind two variables in either object list or action list to a new relation value
    all_vars = list(set(obj_var_list + act_var_list))
    for v1 in all_vars:
        for v2 in all_vars:
            for rel in [0, 1, 2, 3]:  # TODO: this is hard-coded to four for now
                for val in [0, 1]:
                    key = (rel, v1, v2)

                    # continue if the relation is already bound
                    if key in node.relation_bindings:
                        continue

                    # bind the relation to each value
                    relation_bindings = node.relation_bindings.copy()
                    relation_bindings[key] = val
                    counts, classified = check_rule(object_bindings=node.object_bindings,
                                                    action_bindings=node.action_bindings,
                                                    relation_bindings=relation_bindings,
                                                    loader=loader,
                                                    effects=effects,
                                                    dataset_mask=node.dataset_mask)
                    entropy = calculate_entropy(counts)
                    if entropy < best_entropy and np.sum(classified) >= min_samples_split:
                        best_node = Node(left=None, right=None,
                                         object_bindings=node.object_bindings.copy(),
                                         action_bindings=node.action_bindings.copy(),
                                         relation_bindings=relation_bindings,
                                         counts=counts,
                                         dataset_mask=classified)
                        best_entropy = entropy

    return best_entropy, best_node


def learn_tree(loader, effects, min_samples_split=100):
    """Learn a decision tree from the given dataset.

    Args:
        loader (DataLoader): the dataset loader
        effects (np.ndarray): the effects of the dataset
        min_samples_split (int): the minimum number of samples required to split a node

    Returns:
        Node: the root node of the decision tree
    """
    # initialize the root node
    mask = np.zeros(len(loader), dtype=bool)
    root_node = Node(left=None, right=None,
                     object_bindings={},
                     action_bindings={},
                     relation_bindings={},
                     counts=get_effect_counts(effects, mask),
                     dataset_mask=mask)

    # learn the tree
    queue = [root_node]
    num_nodes = 0
    while len(queue) > 0:
        node = queue.pop(0)
        num_nodes += 1
        _, best_node = expand_best_node(node, loader, effects, min_samples_split)
        if best_node is not None:
            print(f"Rule:\n"
                  f"  object bindings={best_node.object_bindings},\n"
                  f"  action bindings={best_node.action_bindings},\n"
                  f"  relation bindings={best_node.relation_bindings},\n"
                  f"  entropy={calculate_entropy(best_node.counts)},\n"
                  f"Num nodes: {num_nodes}")
            node = node._replace(left=best_node)
            right_mask = best_node.dataset_mask.copy()
            right_node = Node(left=None, right=None, object_bindings={}, action_bindings={}, relation_bindings={}, counts=get_effect_counts(effects, right_mask), dataset_mask=right_mask)
            node = node._replace(right=right_node)
            queue.append(node.left)
            queue.append(node.right)
        else:
            print(f"Terminal rule: \n"
                  f"  object bindings={node.object_bindings},\n"
                  f"  action bindings={node.action_bindings},\n"
                  f"  relation bindings={node.relation_bindings},\n"
                  f"  counts={node.counts},\n"
                  f"  entropy={calculate_entropy(node.counts)},\n"
                  f"Num nodes: {num_nodes}")

    return root_node
