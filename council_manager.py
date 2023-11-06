import os
import torch
import wandb
import utils
PATH="./council"

def load(path=PATH):
    council_list = os.listdir("./council")
    assert(len(council_list)> 2)
    council = []
    for m in council_list:
        council.append(torch.load(f"./council/{m}"))
        council[-1].to("cpu")
    return council
def save(council ,path=PATH):
    for i, m in enumerate(council):
        torch.save(m, f"{PATH}/{i}.pt")
def load_experiment(num_generations, experiment):
    api = wandb.Api()
    os.mkdir("./experiments")
    os.mkdir(f"./experiments/{experiment}")
    for i in range(num_generations):
        runs = api.runs(path= "colorslab/active_exploration",filters= {"$and": [{"config.experiment": "single_horizon"}, {"config.generation": f"{i}"}]}, per_page=100)
        for k, run in enumerate(runs):
            model = utils.create_model_from_config(run.config)
            model.load("_best", from_wandb=True, run=run)
            torch.save(model, f"./experiments/{experiment}/generation_{i}_model{k}.pt")


        