import random

import tifffile
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import torchvision

def pad_tensor(tensor, cent, newshape):

    HIGHEST_VALUE = 36863.0
    LOWEST_VALUE = 32995.0

    bgsamp = [33050, 33058, 33058, 33062, 33072, 33066, 33062, 33072, 33066, 33066,
              33056, 33065, 33072, 33072, 33068, 33072, 33081, 33090]
    sig = np.std(np.array(bgsamp))
    mu = np.mean(np.array(bgsamp))

    new_im = (torch.randn(newshape) * sig + mu)
    new_im = (new_im - LOWEST_VALUE) / (HIGHEST_VALUE - LOWEST_VALUE)

    new_im[new_im<0]=0

    tgt_cent = torch.tensor([newshape[0] // 2, newshape[1] // 2])  # (y, x)

    cent=torch.flip(cent.squeeze(),[0])

    cent=(cent+torch.tensor(tensor.shape)/2).to(torch.int)

    dx = tgt_cent - cent  # how much to shift original tensor

    new_im[dx[0]:min(dx[0]+tensor.shape[0],newshape[0]),dx[1]:min(dx[1]+tensor.shape[1],newshape[1])]=tensor

    return new_im




def mass_centroid(tensor):
    # tensor: (B, C, H, W)
    B, C, H, W = tensor.shape

    device = tensor.device

    y_coords = torch.linspace(-H/2,H/2,steps=H, device=device).float()
    x_coords = torch.linspace(-W/2,W/2,steps=W, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)


    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    mass = tensor.sum(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
    mass = mass + 1e-8

    x_centroid = (tensor * x_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)
    y_centroid = (tensor * y_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)

    centroids = torch.stack((x_centroid, y_centroid), dim=-1)
    return centroids


class SingleCellDataset(Dataset):
    HIGHEST_VALUE=36863.0
    LOWEST_VALUE=32995.0
    def __init__(self, dir, target_l,repeat=1):
        # Assuming inputs and targets are lists, convert them to tensors
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.inputs = torch.stack([torch.tensor(i) for i in inputs] * repeat)
        #self.inputs=self.inputs.to(self.device)
        images=[]
        #Loop over images in target directory
        for filename in os.listdir(dir):
            filepath = os.path.join(dir, filename)
            if os.path.isfile(filepath):
                images.append(torch.load(filepath))

        for i,im in enumerate(images):
            # normalize according to max and min in dataset
            im=im.to(torch.float)
            im=(im-self.LOWEST_VALUE)/(self.HIGHEST_VALUE-self.LOWEST_VALUE)
            # Pad to uniform size with plenty of space to transform
            cent=mass_centroid(im[None,None,...])
            im=pad_tensor(im,cent,(target_l,target_l)).view(1,target_l,target_l)
            images[i]=im

        # arrange into tensor and move to CUDA
        self.inputs = torch.stack([torch.tensor(i) for i in images] * repeat)

    def __len__(self):
        return self.inputs.__len__()

    def __getitem__(self, idx):
        return self.inputs[idx]

class LocClassDataset(Dataset):
    def __init__(self, find_dir: list, ignore_dir: list, shape: tuple, noise_mean=0.0, noise_std=1.0):
        self.target_shape = shape  # (H, W)
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.images = []
        self.ignores = []

        # Load and process main images (with padding)
        for dir in find_dir:
            for f in os.listdir(dir):
                if f.lower().endswith(".tif"):
                    path = os.path.join(dir, f)
                    tensor = self._load_and_pad_image(path)
                    self.images.append(tensor)

        # Load and store ignore images (raw, no padding)
        for dir in ignore_dir:
            for f in os.listdir(dir):
                if f.lower().endswith(".tif"):
                    path = os.path.join(dir, f)
                    tensor = self._load_image_raw(path)
                    self.ignores.append(tensor)

        self.pairs = [(i, j) for i in range(len(self.images)) for j in range(len(self.ignores))]
        random.shuffle(self.pairs)


    def _load_and_pad_image(self, path):
        img_np = tifffile.imread(path)
        img_tensor = torch.tensor(img_np, dtype=torch.float32)

        # [C, H, W]
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 3 and img_tensor.shape[0] not in [1, 3]:
            img_tensor = img_tensor.permute(2, 0, 1)

        c, h, w = img_tensor.shape
        th, tw = self.target_shape

        if h > th or w > tw:
            raise ValueError(f"Image at {path} is larger than target shape {self.target_shape}")

        padded = torch.normal(mean=self.noise_mean, std=self.noise_std, size=(c, th, tw))
        padded[padded < 0] = 0

        top = (th - h) // 2
        left = (tw - w) // 2
        padded[:, top:top + h, left:left + w] = img_tensor
        return padded

    def _load_image_raw(self, path):
        img_np = tifffile.imread(path)
        img_tensor = torch.tensor(img_np, dtype=torch.float32)

        # Convert to [C, H, W] if needed
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 3 and img_tensor.shape[0] not in [1, 3]:
            img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.images[i], self.ignores[j]


class VaeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith('.tif')
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = tifffile.imread(self.paths[idx]).astype('float32')
        img = torch.from_numpy(img).unsqueeze(0)  # shape (1, H, W)

        if self.transform:
            img = self.transform(img)

        return img


