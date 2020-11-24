import pickle
from pathlib import Path
from argparse import ArgumentParser
from random import sample, choice
from functools import reduce

import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
# from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_ssim import SSIM, ssim

DATA_ROOT = Path('~/data/FFHQ').expanduser()
LABEL_DIR = 'images256x256'
LATENT_DIR = 'feat_PCA_L5_1024'
N_WORKERS = 4

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
    
def get_dataloaders(root_dir, latent_dir, label_dir,
                    bs_per_gpu, latent_size, data_split=None):
    
    img_list = pickle.loads(Path('img_list.pkl').read_bytes())
    img_list = [root_dir.joinpath(label_dir, *p.parts[1:]) for p in img_list]
    
    latents = []
    latent_paths = sorted(list((root_dir / latent_dir).glob("*.pkl")))
    for p in tqdm(latent_paths):
        latents.append(pickle.loads(p.read_bytes())[:, : latent_size])
    latents = torch.from_numpy(np.concatenate(latents, axis=0))
    assert len(img_list) == latents.shape[0], "#latent & #img are not match"

    if data_split is not None:
        latents_split = []
        imgs_split = []
        idx = 0
        for n_records in data_split:
            latents_split.append(latents[idx:idx+n_records])
            imgs_split.append(img_list[idx:idx+n_records])
            idx += n_records
    else:
        latents_split = [latents, None, None]
        imgs_split = [img_list, None, None]
    
    trf = [
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3, inplace=True),
    ]
    transform = transforms.Compose(trf)
    
    split_name = ['train', 'valid', 'test']
    dataloaders = []
    for i, (split, latents, img_paths) in enumerate(zip(split_name, latents_split, imgs_split)):
        print(f"{split}: ", end="")
        if latents is None or img_paths is None:
            dataloaders.append(None)
            continue
        print(len(img_paths))
        
        dataset = Dataset(latents, img_paths, transforms=transform)
        shuffle = split == 'train'
        dataloaders.append(
            DataLoader(
                dataset,
                bs_per_gpu,
                num_workers=N_WORKERS,
                shuffle=shuffle,
                drop_last=True
            )
        )
    
    return dataloaders

class CNNDecoder(pl.LightningModule):
    def __init__(self, latent_size, batch_size, data_size, log_sample_every, sample_latents, sample_targets, lr=1e-2):
        super(CNNDecoder, self).__init__()
        
        self.lr = lr
        self.save_hyperparameters('latent_size', 'batch_size', 'lr', 'data_size')
        self.linear1 = nn.Linear(latent_size, 4096)
        self.linear2 = nn.Linear(4096, 8*8*256)
        self.act = nn.LeakyReLU(0.2)
        
        n_ch = [256, 256, 128, 128, 64, 64, 3]
        bns = []
        convs = []
        for i in range(len(n_ch) - 1):
            if n_ch[i+1] == 3:
                convs.append(nn.ConvTranspose2d(n_ch[i],n_ch[i+1], kernel_size=1, stride=1, padding=0))
            else:
                convs.append(nn.ConvTranspose2d(n_ch[i],n_ch[i+1], kernel_size=4, stride=2, padding=1))
                
            if i in [4,5]:
                bns.append(nn.BatchNorm2d(n_ch[i+1]))
            else:
                bns.append(None)
            
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Tanh()
        
        # loss
        self.ssim_loss = SSIM()
        self.best_mse = 1.
        self.best_ssim = 0.
        self.log_sample_every = log_sample_every
        
        # sample
        self.register_buffer("sample_latents", sample_latents)
        self.register_buffer("sample_targets", sample_targets)
        assert self.sample_latents.shape[0] == self.sample_targets.shape[0], \
               "sample #latents & #targets are not match"
        self.n_sample = self.sample_latents.shape[0]
    
    def forward(self, latent):
        x = self.linear1(latent)
        x = self.act(x)
        x = self.linear2(x).view(-1,256,8,8)
        
        for conv, bn in zip(self.convs, self.bns):
            x = self.act(x)
            x = conv(x)
            if bn is not None:
                x = bn(x)
        
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        mse_loss, ssim_loss = self.shared_step(batch)
        total_loss = mse_loss
        
        mse_val = mse_loss.item()
        ssim_val = - ssim_loss.item()
        self.log('Metric/MSE', mse_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/SSIM', ssim_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        mse_loss, ssim_loss = self.shared_step(batch)
        mse_val = mse_loss.item()
        ssim_val = - ssim_loss.item()
        self.log('Metric/Val-MSE', mse_val, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/Val-SSIM', ssim_val, on_epoch=True, prog_bar=True, logger=True)
        
        if mse_val < self.best_mse:
            self.best_mse = mse_val
        if ssim_val > self.best_ssim:
            self.best_ssim = ssim_val
            
#         self.logger.log_hyperparams({'batch_size': 1234}, metrics=self.best_ssim)
#         self.logger.log_metrics({'val ssim': ssim_val}, self.global_step)
    
    def shared_step(self, batch):
        latent, target_img = batch
        latent, target_img = latent.cuda(), target_img.cuda()
        fake_img = self(latent)
        
        ssim_loss = - self.ssim_loss((target_img + 1) / 2., (fake_img + 1.) / 2.)
        mse_loss = F.mse_loss(target_img, fake_img)
        
        return mse_loss, ssim_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0, 0.99))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=1000,
            cooldown=200,
            min_lr=1e-4,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Metric/MSE'
        }
        return optimizer, scheduler
        
    def training_epoch_end(self, training_step_outputs):
        if self.current_epoch == 0 or \
            (self.current_epoch+1) % self.log_sample_every == 0:
            self.log_sample_images()
        
    def log_sample_images(self):
        sample_fake_imgs = self(self.sample_latents)
        stack_imgs = torch.stack([self.sample_targets, sample_fake_imgs], dim=0)
        images = stack_imgs.permute(1,0,2,3,4).reshape(self.n_sample*2, *sample_fake_imgs.shape[1:])
        grid_imgs = torchvision.utils.make_grid(images, normalize=True, range=(-1,1), nrow=4)
        self.logger.experiment.add_image('images', grid_imgs, self.current_epoch)
        
    def on_train_start(self):
        print("hyperparams: ", self.hparams)
        self.logger.log_hyperparams({'latent_size': self.lr})
#         self.logger.log_hyperparams([{'bs/gpu': BATCH_SIZE_PER_GPU}, 'something'])
   
def main(hparams):
    ts, vs = hparams.train_size, hparams.val_size
    assert ts + vs <= 70000
    data_split = [ts, vs, 70000-ts-vs]
    
    train_loader, val_loader, test_loader = \
        get_dataloaders(DATA_ROOT, LATENT_DIR, LABEL_DIR, 
                        hparams.batch_size_per_gpu, hparams.latent_size, data_split=data_split)
    latent, target_img = next(iter(train_loader))
    sample_latent_train = latent[:hparams.num_sample]
    sample_img_train = target_img[:hparams.num_sample]
    
    tb_logger = pl_loggers.TensorBoardLogger('runs', name=hparams.log_name, version=hparams.version)
    model = CNNDecoder(hparams.latent_size, hparams.batch_size_per_gpu,
                       hparams.train_size, hparams.log_sample_every,
                       sample_latent_train, sample_img_train, lr=hparams.lr)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         # max_steps=hparams.max_steps,
                         max_epochs=hparams.max_epochs,
                         # resume_from_checkpoint='runs/test/default/version_1/checkpoints/epoch=3.ckpt',
                         logger=tb_logger,
                         sync_batchnorm=True,
                         flush_logs_every_n_steps=10,
                         log_every_n_steps=10,
                         distributed_backend='ddp')
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--max_epochs', type=int, default=1)
#     parser.add_argument('--max_steps', type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=512)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--batch_size_per_gpu', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_sample', type=int, default=32)
    parser.add_argument('--log_name', type=str, default='default')
    parser.add_argument('--log_sample_every', type=int, default=10)
    parser.add_argument('--version', type=str, default=None)

    args = parser.parse_args()

    try:
        args.gpus = int(args.gpus)
    except ValueError:
        pass
    
    main(args)