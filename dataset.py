import os

import torch
from torchvision import transforms

from utils import preprocess

to_tensor = transforms.ToTensor()


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


class SAEFolder(torch.utils.data.Dataset):
    def __init__(self, folder_path, partitions=None):
        self.folder_path = folder_path
        info = list(map(lambda x: int(x.rstrip()), open(os.path.join(folder_path, "info.txt"), "r").readlines()))
        self.sample_per_partition = info[0]
        if partitions is not None:
            self.partitions = partitions
        else:
            self.partitions = list(range(info[1]))
        self.N = self.sample_per_partition * len(self.partitions)

        state = []
        action = []
        effect = []
        for i in self.partitions:
            state.append(torch.load(os.path.join(folder_path, f"state_{i}.pt")))
            action.append(torch.load(os.path.join(folder_path, f"action_{i}.pt")))
            effect.append(torch.load(os.path.join(folder_path, f"effect_{i}.pt")))

        self.state = torch.cat(state)
        self.action = torch.cat(action)
        self.effect = torch.cat(effect)
        assert self.state.shape[0] == self.N
        assert self.action.shape[0] == self.N
        assert self.effect.shape[0] == self.N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        action_idx = self.action[idx]
        eye = torch.eye(6)
        action = torch.cat([eye[action_idx[0]], eye[action_idx[1]]], dim=-1)

        sample = {}
        sample["state"] = self.state[idx] / 255.0
        sample["action"] = action
        sample["effect"] = self.effect[idx] / 255.0 - sample["state"]
        return sample


class CrafterDataset(SAEFolder):
    def __init__(self, folder_path, partitions=None):
        super(CrafterDataset, self).__init__(folder_path, partitions)

    def __getitem__(self, idx):
        sample = {}
        sample["state"] = self.state[idx].flatten(0, 1).float()
        sample["action"] = self.action[idx].float()
        sample["effect"] = self.effect[idx].flatten(0, 1).float()
        sample["pad_mask"] = torch.ones(63)
        return sample


class SegmentedSAEFolder(SAEFolder):
    def __init__(self, folder_path, max_pad, valid_objects, partitions=None, normalize=False, aug=False, old=False, eff_mu=None, eff_std=None):
        super(SegmentedSAEFolder, self).__init__(folder_path, partitions)
        self.max_pad = max_pad
        self.valid_objects = valid_objects
        self.aug = aug
        self.old = old

        segmentation = []
        for i in self.partitions:
            segmentation.append(torch.load(os.path.join(folder_path, f"segmentation_{i}.pt")))
        self.segmentation = torch.cat(segmentation)
        assert self.segmentation.shape[0] == self.N

        if normalize:
            effect_shape = self.effect.shape
            effect = self.effect.flatten(0, -2)
            if (eff_mu is not None) and (eff_std is not None):
                self.eff_mu = eff_mu
                self.eff_std = eff_std
            else:
                self.eff_mu = effect.mean(dim=0)
                self.eff_std = effect.std(dim=0) + 1e-6
            effect = (effect - self.eff_mu) / (self.eff_std)
            self.effect = effect.reshape(effect_shape)

    def __getitem__(self, idx):
        padded, pad_mask = preprocess(self.state[idx], self.segmentation[idx], self.valid_objects, self.max_pad, self.aug, self.old)
        action_idx = self.action[idx]
        eye = torch.eye(6)
        action = torch.cat([eye[action_idx[0]], eye[action_idx[1]]], dim=-1)
        action = action.unsqueeze(0)
        action = action.repeat(padded.shape[0], 1)

        sample = {}
        sample["state"] = padded
        sample["action"] = action
        sample["effect"] = self.effect[idx][..., :3]
        sample["pad_mask"] = pad_mask
        return sample


class SegmentedSAEFolder3x5(SegmentedSAEFolder):
    def __init__(self, folder_path, max_pad, valid_objects, partitions=None, normalize=False, aug=False, old=False, eff_mu=None, eff_std=None):
        super(SegmentedSAEFolder3x5, self).__init__(folder_path, max_pad, valid_objects, partitions, normalize, aug, old, eff_mu, eff_std)

    def __getitem__(self, idx):
        padded, pad_mask = preprocess(self.state[idx], self.segmentation[idx], self.valid_objects, self.max_pad, self.aug, self.old)
        action_idx = self.action[idx]
        eye_3, eye_5 = torch.eye(3), torch.eye(5)
        action = torch.cat([eye_3[action_idx[0]], eye_5[action_idx[1]], eye_3[action_idx[2]], eye_5[action_idx[3]]], dim=-1)
        action = action.unsqueeze(0)
        action = action.repeat(padded.shape[0], 1)

        sample = {}
        sample["state"] = padded
        sample["action"] = action
        sample["effect"] = self.effect[idx][..., :3]
        sample["pad_mask"] = pad_mask
        return sample
