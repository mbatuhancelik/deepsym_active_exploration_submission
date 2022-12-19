import wandb
import argparse
import torch
import utils
from dataset import StateActionEffectDataset



parser = argparse.ArgumentParser("Generate Z and E values for the dataset.")
parser.add_argument("-id", help="experiment id")
parser.add_argument("-db", help="dataset folder name under ./data")
args = parser.parse_args()


run = wandb.init(entity="colorslab",project="multideepsym", resume="must", id=args.id)
run = wandb.Api().run("colorslab/multideepsym/" + args.id)
run.config["device"] = "cpu"


model = utils.create_model_from_config(run.config)
model.load("_best", from_wandb=True)

dataset = StateActionEffectDataset(args.db, split="test")
 

state_pad = torch.zeros(size=(1,13,1), device='cpu')
Z = []
E = []
Z_prime = []
for i in range(len(dataset)):
    with torch.no_grad():
        data = dataset[i]
        for key in ["action", "state", "effect"]:
            data[key] = data[key].unsqueeze(0).expand(-1, 13, -1)

        data["pad_mask"] = data["pad_mask"].unsqueeze(0).expand(-1, 13)

        z , e = model.forward(data, eval_mode=True)
        data["state"] += torch.cat([e, state_pad],dim=-1).to('cpu')
        z_prime , _ = model.forward(data,eval_mode= True)
        Z.append(z.to("cpu"))
        E.append(e.to("cpu"))
        Z_prime.append(z_prime.to("cpu"))
        if(i %1000 == 0):
            print(i, "processed")

Z = torch.cat(Z, axis = 0)
Z_prime = torch.cat(Z_prime, axis = 0)
E = torch.cat(E, axis = 0)

folder = "./wandb/latest-run/files/save/deneme/"

torch.save(Z, folder + "Z.pt")
torch.save(Z_prime, folder + "Z_prime.pt")
torch.save(E, folder + "E.pt")

wandb.save(folder + "Z.pt")
wandb.save(folder + "Z_prime.pt")
wandb.save(folder + "E.pt")



        
# run = wandb.init(id="e16396qx")
# artifact = run.use_artifact('colorslab/multideepsym/v4:v0', type='dataset')
# artifact_dir = artifact.download()
# run = wandb.Api().run("colorslab/multideepsym/e16396qx")

