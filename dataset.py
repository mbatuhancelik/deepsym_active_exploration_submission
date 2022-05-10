import os
from PIL import Image

import torch
from torchvision import transforms

from utils import segment_img_with_mask

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
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.N = int(open(os.path.join(folder_path, "info.txt"), "r").read().rstrip())
        self.action = torch.load(os.path.join(folder_path, "action.pt"))
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        path_a = os.path.join(self.folder_path, "%d_a_rgb.png" % idx)
        path_b = os.path.join(self.folder_path, "%d_b_rgb.png" % idx)
        img_a = to_tensor(Image.open(path_a))
        img_b = to_tensor(Image.open(path_b))
        sample = {}
        sample["state"] = img_a
        sample["action"] = self.action[idx].float()
        sample["effect"] = img_b - img_a
        return sample


class SegmentedSAEFolder(SAEFolder):
    def __init__(self, folder_path, max_pad):
        super(SegmentedSAEFolder, self).__init__(folder_path)
        self.max_pad = max_pad

    def __getitem__(self, idx):
        path_a = os.path.join(self.folder_path, "%d_a_rgb.png" % idx)
        path_b = os.path.join(self.folder_path, "%d_b_rgb.png" % idx)
        img_a = to_tensor(Image.open(path_a))
        img_b = to_tensor(Image.open(path_b))
        mask_a = torch.load(os.path.join(self.folder_path, "%d_a_seg.pt" % idx))
        seg_a = segment_img_with_mask(img_a, mask_a)
        n_seg, ch, h, w = seg_a.shape
        n_seg = min(n_seg, self.max_pad)
        padded = torch.zeros(self.max_pad, ch, h, w)
        padded[:n_seg] = seg_a[:n_seg]
        pad_mask = torch.zeros(self.max_pad)
        pad_mask[:n_seg] = 1.0

        sample = {}
        sample["state"] = padded
        sample["action"] = self.action[idx].float()
        sample["effect"] = img_b - img_a
        sample["pad_mask"] = pad_mask
        return sample
