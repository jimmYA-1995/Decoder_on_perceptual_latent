import pickle
from pathlib import Path
from argparse import ArgumentParser
from random import sample, choice
from functools import reduce

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
# from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from models import LitSystem


class Dataset(torch.utils.data.Dataset):
    def __init__(self, latents, img_paths, transforms=None):
        assert len(img_paths) == latents.shape[0]
        self.latents = latents
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return self.latents.shape[0]
    
    def __getitem__(self, index):
        latent = self.latents[index]
        target_img = io.imread(self.img_paths[index])
        
        if self.transforms:
            target_img = self.transforms(target_img)
            
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
    assert len(img_list) == latents.shape[0], f"#latent & #img are not match {latents.shape[0]} v.s. {len(img_list)}"
    assert sum(data_split) <= len(img_list)
    
    trf = [
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3, inplace=True),
    ]
    trfs = transforms.Compose(trf)
     
    idx = 0
    dataloaders = []   
    split_names = ['train', 'val', 'test']
    for split, n_records in zip(split_names, data_split):
        l = latents[idx:idx+n_records]
        img_paths = img_list[idx:idx+n_records]
        idx += n_records
        
        dataset = Dataset(l, img_paths, transforms=trfs)
        dataloaders.append(
            DataLoader(
                dataset,
                bs_per_gpu,
                num_workers=num_workers,
                shuffle=(split == 'train'),
                drop_last=True
            )
        )
    
    return dataloaders


def main(args):
    data_split = [args.train_size, args.val_size, args.test_size]
    
    train_loader, val_loader, test_loader = \
        get_dataloaders(args.root_dir, args.latent_path, args.target_dir, data_split,
                        args.num_workers, args.latent_dim, args.bs_per_gpu)
    
    samples = {}
    latent, target_img = next(iter(train_loader))
    samples['train'] = {
        'latents': latent[:args.num_sample],
        'targets': target_img[:args.num_sample]
    }
    latent, target_img = next(iter(test_loader))
    samples['test'] = {
        'latents': latent[:args.num_sample],
        'targets': target_img[:args.num_sample]
    }

    train_system = LitSystem(args.log_sample_every,
                             samples,
                             lr=args.lr,
                             lr_scheduler=args.lr_scheduler,
                             batch_size=(args.n_gpu*args.bs_per_gpu),
                             norm_type=args.norm_type,
                             latent_dim=args.latent_dim,
                             train_size=args.train_size,
                             val_size=args.val_size)
    tb_logger = pl_loggers.TensorBoardLogger('runs', name=args.log_name, version=args.version, default_hp_metric=False)
    kwargs = dict(logger=tb_logger, distributed_backend='ddp', automatic_optimization=False)
    if not args.resume_from_checkpoint:
        kwargs.update({
            'resume_from_checkpoint': args.resume_from_checkpoint
        })
    
    trainer = Trainer.from_argparse_args(args, **kwargs)
    trainer.fit(train_system, train_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # program arguments
    DATA_ROOT = Path('~/data/FFHQ').expanduser()
    
#     parser.add_argument('--max_steps', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=32)
    parser.add_argument('--log_name', type=str, default='default')
    parser.add_argument('--version', type=str, default=None)

    # data
    parser.add_argument('--root_dir', type=str, default='~/data/FFHQ')
    parser.add_argument('--latent_path', type=str, default='feat_PCA_L5_1024')
    parser.add_argument('--target_dir', type=str, default='images256x256')

    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--train_size', type=int, default=3200)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--bs_per_gpu', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--norm_type', type=str, default='batch_norm')
    
    parser = LitSystem.add_model_specific_args(parser)
    
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    # gpus, max_epochs, (max_steps),
    # resume_from_checkpoints, sync_batchnorm
    # log_every_n_steps, flush_logs_every_n_steps
    # distributed_backend

    args = parser.parse_args()

    try:
        args.gpus = int(args.gpus)
        args.n_gpu = args.gpus
    except ValueError:
        args.n_gpu = len(args.gpus.split(','))
    
    main(args)
