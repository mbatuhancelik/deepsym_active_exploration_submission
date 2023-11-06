import os
import yaml
import argparse

def load(path): 
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save(path, config):
    yaml.dump(config,open(path, "w"))

def set_generation(config, gen):
    config["generation"] =gen
    return config
def set_dataset(config, dataset):
    assert(os.path.isdir(f"./data/{dataset}"))
    config["dataset_name"] = dataset
    return config
def set_experiment_name(config, exp):
    config["experiment"] = exp
    return config

if __name__ == "__main__":

    parser = argparse.ArgumentParser("See dataset metrics.")
    parser.add_argument("action", type=str)
    parser.add_argument("-c", help="config path", type=str, required=True)
    parser.add_argument("-d", help="dataset name", type=str)
    parser.add_argument("-e", help="experiment name", type=str)
    parser.add_argument("-g", help="generation", type=str)

    args = parser.parse_args()

    config = load(args.c)

    args = parser.parse_args()
    if args.action == "set_generation":
        config = set_generation(config, args.g)
    if args.action == "set_dataset":
        config = set_dataset(config, args.d)
    if args.action == "set_experiment":
        config = set_experiment_name(config, args.e)
    save(args.c, config)

