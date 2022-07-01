import os

import torch
from torchvision import transforms

from utils import segment_img_with_mask, segment_img_with_mask_old

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
    def __init__(self, folder_path, num_partition=None):
        self.folder_path = folder_path
        info = list(map(lambda x: int(x.rstrip()), open(os.path.join(folder_path, "info.txt"), "r").readlines()))
        self.sample_per_partition = info[0]
        if num_partition is not None:
            self.num_partition = num_partition
        else:
            self.num_partition = info[1]
        self.N = self.sample_per_partition * self.num_partition

        state = []
        action = []
        effect = []
        # segmentation = []
        for i in range(self.num_partition):
            state.append(torch.load(os.path.join(folder_path, f"state_{i}.pt")))
            action.append(torch.load(os.path.join(folder_path, f"action_{i}.pt")))
            effect.append(torch.load(os.path.join(folder_path, f"effect_{i}.pt")))
            # segmentation.append(torch.load(os.path.join(folder_path, f"segmentation_{i}.pt")))

        self.state = torch.cat(state)
        self.action = torch.cat(action)
        self.effect = torch.cat(effect)
        # self.segmentation = torch.cat(segmentation)
        assert self.state.shape[0] == self.N
        assert self.action.shape[0] == self.N
        assert self.effect.shape[0] == self.N
        # assert self.segmentation.shape[0] == self.N

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
    def __init__(self, folder_path, num_partition=None):
        super(CrafterDataset, self).__init__(folder_path, num_partition)

    def __getitem__(self, idx):
        sample = {}
        sample["state"] = self.state[idx].flatten(0, 1).float()
        sample["action"] = self.action[idx].float()
        sample["effect"] = self.effect[idx].flatten(0, 1).float()
        sample["pad_mask"] = torch.ones(63)
        return sample


class SegmentedSAEFolder(SAEFolder):
    def __init__(self, folder_path, max_pad, valid_objects, num_partition=None, normalize=False, aug=False, old=False):
        super(SegmentedSAEFolder, self).__init__(folder_path, num_partition)
        self.max_pad = max_pad
        self.valid_objects = valid_objects
        self.aug = aug
        self.old = old

        segmentation = []
        for i in range(self.num_partition):
            segmentation.append(torch.load(os.path.join(folder_path, f"segmentation_{i}.pt")))
        self.segmentation = torch.cat(segmentation)
        assert self.segmentation.shape[0] == self.N

        if normalize:
            effect_shape = self.effect.shape
            effect = self.effect.flatten(0, -2)
            self.eff_mu = effect.mean(dim=0)
            self.eff_std = effect.std(dim=0)
            effect = (effect - self.eff_mu) / (self.eff_std + 1e-6)
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


def preprocess(state, segmentation, valid_objects, max_pad, aug, old):
    if old:
        seg_a = segment_img_with_mask_old(state, segmentation, valid_objects)
    else:
        seg_a = segment_img_with_mask(state, segmentation, valid_objects, aug=aug)
    n_seg, ch, h, w = seg_a.shape
    n_seg = min(n_seg, max_pad)
    padded = torch.zeros(max_pad, ch, h, w)
    padded[:n_seg] = seg_a[:n_seg]
    pad_mask = torch.zeros(max_pad)
    pad_mask[:n_seg] = 1.0
    return padded, pad_mask
