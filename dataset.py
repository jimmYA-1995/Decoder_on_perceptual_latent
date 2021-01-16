import pickle
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
from tqdm import tqdm
from skimage.transform import resize
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

class Dataset(data.Dataset):
    def __init__(self, latents, img_paths, transforms=None, return_indices=False):
        assert len(img_paths) == latents.shape[0]
        self.latents = latents
        self.latent_indices = torch.arange(len(latents), dtype=torch.long)
        self.img_paths = img_paths
        self.transforms = transforms
        self.return_indices = return_indices

    def __len__(self):
        return self.latents.shape[0]
    
    def __getitem__(self, index):
        latent = self.latents[index]
        latent_indices = self.latent_indices[index]
        target_img = io.imread(self.img_paths[index])
        
        if self.transforms:
            target_img = self.transforms(target_img)
        
        if self.return_indices:
            return latent_indices, latent, target_img
        return latent, target_img

def get_dataloaders(root_dir, latent_path, target_dir, data_split,
                    num_workers=1, latent_dim=512, bs_per_gpu=32):
    root_dir = Path(root_dir).expanduser() \
               if root_dir.startswith('~') \
               else Path(root_dir)
    img_list = sorted(list((root_dir / target_dir).glob('*/*.png')))

    latents = []
    if Path(latent_path).is_dir():
        latent_paths = sorted(list((root_dir / latent_path).glob("*.pkl")))
        for p in tqdm(latent_paths):
            latents.append(pickle.loads(p.read_bytes())[:, :latent_dim])
        latents = torch.from_numpy(np.concatenate(latents, axis=0))
    elif Path(latent_path).suffix == ".npy":
        latents = torch.from_numpy(np.load(root_dir / latent_path)[:, :latent_dim])
    else:
        raise NotImplementedError("feat format not supported")

    if latents.dtype == torch.float64:
        latents = latents.float()
    assert len(img_list) == latents.shape[0], \
           f"#latent & #img are not match {latents.shape[0]} v.s. {len(img_list)}"
    assert sum(data_split) <= len(img_list)
    
    trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3, inplace=True),
    ])
     
    idx = 0
    dataloaders = []   
    split_names = ['train', 'val', 'test']
    for split, n in zip(split_names, data_split):
        if n == 0:
            dataloaders.append(None)
            continue
        
        l = latents[idx:idx+n]
        if split == 'train':
            train_latents = l.clone()
                                    
        img_paths = img_list[idx:idx+n]
        idx += n
        
        dataset = Dataset(l, img_paths, transforms=trfs, return_indices=(split == 'train'))
        dataloaders.append(
            DataLoader(
                dataset,
                bs_per_gpu,
                num_workers=num_workers,
                shuffle=(split == 'train'),
                drop_last=True
            )
        )
    
    return dataloaders, train_latents