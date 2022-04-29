import os

import torch


class StateActionDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.state = torch.load(os.path.join(path, "state.pt"))
        self.action = torch.load(os.path.join(path, "action.pt"))

    def __len__(self):
        return len(self.state)-1

    def __getitem__(self, idx):
        sample = {}
        sample["state"] = self.state[idx] / 255.0
        sample["state_n"] = self.state[idx+1] / 255.0
        sample["action"] = self.action[idx].float()
        sample["effect"] = sample["state_n"] - sample["state"]
        return sample


class StateActionEffectDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.state = torch.load(os.path.join(path, "state.pt"))
        self.action = torch.load(os.path.join(path, "action.pt"))
        self.effect = torch.load(os.path.join(path, "effect.pt"))

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        sample = {}
        sample["state"] = self.state[idx] / 255.0
        sample["action"] = self.action[idx].float()
        sample["effect"] = self.effect[idx] / 255.0
        return sample
