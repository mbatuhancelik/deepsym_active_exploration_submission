import os
import torch
import wandb
import utils
import argparse
PATH="./council"

def load(path=PATH):
    council_list = os.listdir("./council")
    assert(len(council_list)> 2)
    council = []
    for m in council_list:
        council.append(torch.load(f"./council/{m}"))
        council[-1].to("cpu")
    return council
def get_avg_loss(experiment, generation):
    api = wandb.Api()
    runs = api.runs(path= "colorslab/active_exploration",filters= {"$and": [{"config.experiment": experiment}, {"config.generation": generation}]}, per_page=100)
    run_ids = [run.id for run in runs[:max(4, int((len(runs)+1)//2))]]
    print(len(runs), runs[:int((len(runs)+1)//2)])
    return wandb.run.id in run_ids
def save(council ,path=PATH):
    for i, m in enumerate(council):
        torch.save(m, f"{PATH}/{i}.pt")
def load_experiment(num_generations, experiment):
    api = wandb.Api()
    try:
        os.mkdir("./experiments")
        os.mkdir(f"./experiments/{experiment}")
    finally:
        pass
    for i in range(num_generations):
        runs = api.runs(path= "colorslab/active_exploration",filters= {"$and": [{"config.experiment": experiment}, {"config.generation": f"{i}"}]}, per_page=100)
        for k, run in enumerate(runs):
            model = utils.create_model_from_config(run.config)
            model.load("_best", from_wandb=True, run=run)
            torch.save(model, f"./experiments/{experiment}/generation_{i}_model{k}.pt")
def load_generation(experiment, generation, folder, take_half=False):
        api = wandb.Api()
        runs = api.runs(path= "colorslab/active_exploration",
                        filters= {"$and": [{"config.experiment": experiment}, {"config.generation": f"{generation}"}]}, 
                        per_page=100, 
                        order="+summary_metrics.best_val_loss")
        if take_half:
            runs = runs[:5]
        if not os.path.exists(folder):
            os.makedirs(folder)
        for k, run in enumerate(runs):
            model = utils.create_model_from_config(run.config)
            model.load("_best", from_wandb=True, run=run)
            model.to("cpu")
            torch.save(model, f"{folder}/generation_{generation}_model_{k}_{run.id}.pt")
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Load Council.")
    parser.add_argument("experiment", type=str)
    parser.add_argument("generation", type=int)
    parser.add_argument("folder", type=str)
    parser.add_argument("take_half", type=bool)

    args = parser.parse_args()
    for i in range(args.generation):
        load_generation(args.experiment, str(i), args.folder, args.take_half)

        