from argparse import ArgumentParser
from typing import List
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule

from pytorch_ssim import SSIM, ssim
import lpips
from .cnn_decoder import CNNDecoder


class LitSystem(LightningModule):
    def __init__(self,
                 decoder: LightningModule,
                 log_sample_every,
                 samples,
                 losses: List[str] = ['mse'],
                 lr: float = 0.2,
                 batch_size: int = 32,
                 latent_dim: int = 512,
                 train_size: int = 3200,
                 norm_type: str = 'batch_norm',
                 val_size: int = 3200,
                 val_MSE: float = -1.,
                 val_SSIM: float = -1.,
                 val_LPIPS: float = -1.):
        
        super(LitSystem, self).__init__()
        
        # medium tutorial: https://medium.com/pytorch/pytorch-lightning-metrics-35cb5ab31857
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2406
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4030#issuecomment-708274317
        # register hyparams & metric in the init. and update value later
        self.save_hyperparameters("train_size", "val_size", "lr", "batch_size", "latent_dim", "norm_type", "val_MSE", "val_SSIM", "val_LPIPS")
        self.decoder = decoder
        self.lr = lr
        
        # loss
        self.ssim_loss = SSIM()
        self.percept = lpips.PerceptualLoss(
            model='net-lin', net='alex', use_gpu=True
        )
        for param in percept.model.net.parameters():
            param.requires_grad = False
        
        # metric & log
        self.best_mse = 4.
        self.best_ssim = 0.
        self.best_lpips = float("inf")
        self.log_sample_every = log_sample_every
        
        # sample
        self.register_buffer("smpl_latent_train", samples['train']['latents'])
        self.register_buffer("smpl_target_train", samples['train']['targets'])
        self.register_buffer("smpl_latent_test", samples['test']['latents'])
        self.register_buffer("smpl_target_test", samples['test']['targets'])
        self.n_sample = self.smpl_latent_train.shape[0]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--', type=, default=, help='')
        parser.add_argument('--lr', type=float, default=1e-2)
        # parser.add_argument('--optim', type=str, default='Adam')
        # parser.add_argument('--beta_1', type=float, default=0.)
        # parser.add_argument('--beta_2', type=float, default=0.99)
        # parser.add_argument('--lr_scheduler', type=str, default=None, choice=['ReduceLROnPlateau'])
        parser.add_argument('--log_sample_every', type=int, default=10)
        
        return parser
        
    
    def forward(self, latent):
        return self.decoder(latent)

    def shared_step(self, batch):
        latent, target_img = batch
        # latent, target_img = latent.cuda(), target_img.cuda()
        fake_img = self.decoder(latent)
        
        ssim_loss = - self.ssim_loss((target_img + 1) / 2., (fake_img + 1.) / 2.)
        mse_loss = F.mse_loss(target_img, fake_img)
        lpips_loss = self.percept((target_img + 1) / 2., (fake_img + 1.) / 2.)
        mse_val = mse_loss.detach()
        ssim_val = - ssim_loss.detach()
        lpips_val = lpips_loss.detach()
        
        return mse_loss, ssim_loss, lpips_loss, mse_val, ssim_val, lpips_val

    def on_train_start(self):
        pass
#         print("hyperparams: ", self.hparams)
#         self.logger.log_hyperparams({'latent_dim': self.lr})
#         self.logger.log_hyperparams([{'bs/gpu': bs_per_gpu}, 'something'])

    def training_step(self, batch, batch_idx):
        mse_loss, ssim_loss, lpips_loss, mse_val, ssim_val, lpips_val = self.shared_step(batch)
        total_loss = mse_loss 

        self.log('Metric/MSE', mse_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/SSIM', ssim_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/LPIPS', lpips_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss
    
    def training_epoch_end(self, training_step_outputs):
        if self.current_epoch == 0 or \
            (self.current_epoch+1) % self.log_sample_every == 0:
            self.log_sample_images()
            self.log_sample_images(mode='test')
            self.log_interpolated_images()
            
    def validation_step(self, batch, batch_idx):
        mse_loss, ssim_loss, lpips_loss, mse_val, ssim_val, lpips_val = self.shared_step(batch)
        self.log('Metric/Val-MSE', mse_val, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/Val-SSIM', ssim_val, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/LPIPS', lpips_val, on_epoch=True, prog_bar=True, logger=True)
        
        return {'mse': mse_val, 'ssim': ssim_val, 'lpips': lpips_val}
    
    def validation_epoch_end(self, validation_step_outputs):
        epoch_mse = validation_step_outputs[0]['mse'].mean()
        epoch_ssim = validation_step_outputs[0]['ssim'].mean()
        epoch_lpips = validation_step_outputs[0]['lpips'].mean()
        if epoch_mse < self.best_mse:
            self.best_mse = epoch_mse
        if epoch_ssim > self.best_ssim:
            self.best_ssim = epoch_ssim
        if epoch_lpips < self.best_lpips:
            self.best_lpips = epoch_lpips
        
        self.logger.log_hyperparams(params=self.hparams,
                                    metrics={
                                        'val_MSE': self.best_mse,
                                        'val_SSIM': self.best_ssim,
                                        'val_LPIPS': self.best_lpips
                                    })
        
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
        
    def log_sample_images(self, mode='train'):
        sample_fake_imgs = self(getattr(self, f'smpl_latent_{mode}'))
        stack_imgs = torch.stack([getattr(self, f'smpl_target_{mode}'), sample_fake_imgs], dim=0)
        images = stack_imgs.permute(1,0,2,3,4).reshape(self.n_sample*2, *sample_fake_imgs.shape[1:])
        grid_imgs = torchvision.utils.make_grid(images, normalize=True, range=(-1,1), nrow=4)
        self.logger.experiment.add_image(f'{mode}_imgs', grid_imgs, self.current_epoch)
        
    def log_interpolated_images(self):
        N_INTERPOLATION = 10
        latents = self.smpl_latent_train
        
        fake_img_list = []
        for idx in range(0, latents.shape[0], 2):
            latent_e1 = latents[idx]
            latent_e2 = latents[idx+1]
            step = (latent_e2 - latent_e1) / N_INTERPOLATION
            latent_list = list(map(lambda i: latent_e1 + i * step, range(1, N_INTERPOLATION)))
            latent_list = [latent_e1] + latent_list + [latent_e2]
            latent_list = [x.unsqueeze(0) for x in latent_list]
            batch_latent = torch.cat(latent_list, axis=0)

            with torch.no_grad():
                fake_imgs = self(batch_latent)
                res = fake_imgs.shape[2]
                fake_imgs = fake_imgs.permute(0,3,2,1)
                fake_imgs = fake_imgs.reshape(res*batch_latent.shape[0], res, 3)
                fake_imgs = fake_imgs.permute(2,1,0)
                fake_img_list.append(fake_imgs)
                
        interpolated_gallery = torch.cat(fake_img_list, dim=1).cpu().numpy()
        interpolated_gallery = (interpolated_gallery * 127.5 + 127.5).astype(np.uint8)
        self.logger.experiment.add_image('interpolated_img', interpolated_gallery, self.current_epoch)        