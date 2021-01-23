import random
from statistics import mean
from typing import List
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule

from pytorch_ssim import SSIM, ssim
from lpips_pytorch import LPIPS
from .cnn_decoder import CNNDecoder


class LitSystem(LightningModule):
    def __init__(self,
                 decoder: LightningModule,
                 log_sample_every,
                 samples,
                 losses: str = 'mse',
                 lr: float = 0.2,
                 lr_scheduler: str = None,
                 batch_size: int = 32,
                 latent_dim: int = 512,
                 train_size: int = 3200,
                 norm_type: str = 'batch_norm',
                 val_size: int = 3200,
                 val_MSE: float = -1.,
                 val_SSIM: float = -1.,
                 val_LPIPS: float = -1.,
                 val_tri_neq: float = -1):
        
        super(LitSystem, self).__init__()
        
        # medium tutorial: https://medium.com/pytorch/pytorch-lightning-metrics-35cb5ab31857
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2406
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4030#issuecomment-708274317
        # register hyparams & metric in the init. and update value later
        self.save_hyperparameters("train_size", "val_size", "losses", "lr", "batch_size", "latent_dim", "norm_type", "val_MSE", "val_SSIM", "val_LPIPS", "val_tri_neq")
        self.decoder = decoder
        self.losses = losses.split(',')
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        
        # loss
        self.ssim_loss = SSIM()
        # https://github.com/S-aiueo32/lpips-pytorch
        # not using official repo. because we can't use sub-DataParallel model in pytorh-lightning
        self.percept = LPIPS(
            net_type='alex',
            version='0.1'
        )
        for param in self.percept.parameters():
            param.requires_grad = False
        
        # metric & log
        self.best_mse = 4.
        self.best_ssim = 0.
        self.best_lpips = float("inf")
        self.best_tri_neq = float("inf")
        self.log_sample_every = log_sample_every
        
        # sample
        self.register_buffer("smpl_indices_train", samples['train']['indices'])
        self.register_buffer("smpl_latent_train", samples['train']['latents'])
        self.register_buffer("smpl_target_train", samples['train']['targets'])
        self.register_buffer("smpl_latent_test", samples['test']['latents'])
        self.register_buffer("smpl_target_test", samples['test']['targets'])
        self.n_sample = self.smpl_latent_train.shape[0]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-2)
        # parser.add_argument('--optim', type=str, default='Adam')
        # parser.add_argument('--beta_1', type=float, default=0.)
        # parser.add_argument('--beta_2', type=float, default=0.99)
        # parser.add_argument('--use_tri_neq', type=bool, default=False, action='store_true', help='whether using tri inequality on lpips')
        parser.add_argument('--losses', type=str, default='mse', help='comma seperated str. e.g. mse,lpips,ssim')
        parser.add_argument('--lr_scheduler', type=str, choices=['None', 'ReduceLROnPlateau', 'MultiStepLR'])
        parser.add_argument('--log_sample_every', type=int, default=10)
        
        return parser
        
    def forward(self, latent, indices=None):
        return self.decoder(latent, indices=indices)

    def shared_step(self, latent, target_img, indices=None, reg=False):
        tri_ineq_reg, tri_neq_val = None, None

        b, c, h, w = target_img.shape        
        
        fake_imgs_e = self.decoder(latent, indices=indices)
        mse_loss = F.mse_loss(target_img, fake_imgs_e)
        ssim_loss = -self.ssim_loss((target_img + 1) / 2., (fake_imgs_e + 1.) / 2.)
        lpips_loss = self.percept((target_img + 1) / 2., (fake_imgs_e + 1.) / 2.).mean()

        losses = dict(mse=mse_loss, ssim=ssim_loss, lpips=lpips_loss)
        if reg:
            _, nz = latent.shape
            latent_l, latent_r = latent.view(2, b//2, nz)
            fake_imgs_l, fake_imgs_r = fake_imgs_e.view(2, b//2, c, h, w)
            fake_imgs_l, fake_imgs_r = fake_imgs_l.detach(), fake_imgs_r.detach()
            
            alpha = torch.rand((b//2,1)).type_as(latent)
            interpolated_latents = latent_l * alpha + latent_r * (torch.ones_like(alpha) - alpha)
            fake_imgs_c = self.decoder(interpolated_latents, indices=indices)

            lpips_lr = self.percept((fake_imgs_l + 1) / 2., (fake_imgs_r + 1.) / 2.)
            lpips_cl = self.percept((fake_imgs_c + 1) / 2., (fake_imgs_l + 1.) / 2.)
            lpips_cr = self.percept((fake_imgs_c + 1) / 2., (fake_imgs_r + 1.) / 2.)

            assert (lpips_cl >= 0).all() and (lpips_cr >= 0).all() and (lpips_lr >= 0).all(), "lpips small than zero"
            #if ((lpips_cl + lpips_cr - lpips_lr) >= 0).all():
            #    print(f"not follow triangle inequality.\n {lpips_lr[:,0,0,0]}\n {lpips_cr[:,0,0,0]}\n {lpips_cl[:,0,0,0]}")
            tri_ineq_reg = (lpips_cl + lpips_cr) / lpips_lr - 1
            losses['tri_ineq'] = tri_ineq_reg.mean()            

        return losses

    def on_train_start(self):
        embed = self.decoder.embed.weight.detach().cpu().numpy()
        np.save('embed_init', embed)

    def training_step(self, batch, batch_idx):
        indices, latent, target_img = batch
        
        use_reg = True if self.current_epoch > 10 and batch_idx%10 == 0 else False ##
        losses = self.shared_step(latent, target_img, indices=indices,reg=use_reg)
        
        total_loss = sum([v for k,v in losses.items() if k in self.losses])

        self.log('Metric/MSE', losses['mse'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/SSIM', -losses['ssim'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/LPIPS', losses['lpips'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if use_reg:
            total_loss = total_loss + 0.1 * losses['tri_ineq']
            self.log('Metric/tri-neq', losses['tri_ineq'].item(),
                     on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        embed = self.decoder.embed.weight
        norms = torch.linalg.norm(embed, dim=1)
        embed_mean, embed_std = norms.mean().item(), norms.std().item()
        norms_l1 = torch.linalg.norm(self.decoder.linear1.weight).item()
        norms_l2 = torch.linalg.norm(self.decoder.linear2.weight).item()
        self.log('Stats/EmbedNorm-Mean', embed_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Stats/EmbedNorm-Std', embed_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Stats/Linear1Norm', norms_l1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Stats/Linear2Norm', norms_l2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss
    
    def training_epoch_end(self, training_step_outputs):
        if self.current_epoch == 0 or \
            (self.current_epoch+1) % self.log_sample_every == 0:
            self.log_sample_images()
            self.log_sample_images(mode='test')
            self.log_interpolated_images()
            
    def validation_step(self, batch, batch_idx):
        latent, target_img = batch
        losses = self.shared_step(latent, target_img, reg=True)
        metrics = {
            'mse': losses['mse'].item(),
            'ssim': -losses['ssim'].item(),
            'lpips': losses['lpips'].item(),
            'tri_neq': losses['tri_ineq'].item()
        }
        # TODO: sync_dist=True
        self.log('Metric/Val-MSE', metrics['mse'], on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/Val-SSIM', metrics['ssim'], on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/Val-LPIPS', metrics['lpips'], on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/Val-tri-neq', metrics['tri_neq'], on_epoch=True, prog_bar=True, logger=True)
        
        return metrics
    
    def validation_epoch_end(self, validation_step_outputs):
        epoch_mse = mean([x['mse'] for x in validation_step_outputs])
        epoch_ssim = mean([x['ssim'] for x in validation_step_outputs])
        epoch_lpips = mean([x['lpips'] for x in validation_step_outputs])
        epoch_tri_neq = mean([x['tri_neq'] for x in validation_step_outputs])
        if epoch_mse < self.best_mse:
            self.best_mse = epoch_mse
        if epoch_ssim > self.best_ssim:
            self.best_ssim = epoch_ssim
        if epoch_lpips < self.best_lpips:
            self.best_lpips = epoch_lpips
        if epoch_tri_neq < self.best_tri_neq:
            self.best_tri_neq = epoch_tri_neq
        
        self.logger.log_hyperparams(params=self.hparams,
                                    metrics={
                                        'val_MSE': self.best_mse,
                                        'val_SSIM': self.best_ssim,
                                        'val_LPIPS': self.best_lpips,
                                        'val_tri_neq': self.best_tri_neq
                                    })
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr, betas=(0, 0.99))
        scheduler = None
        if self.lr_scheduler != 'None':
            if self.lr_scheduler == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=1000,
                    cooldown=200,
                    min_lr=1e-4,
                    verbose=True
                )
            elif self.lr_scheduler == 'MultiStepLR':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,110,140], gamma=0.1)
            else:
                raise NotImplementedError("Learning rate scheduler type not supported yet")

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'Metric/MSE'
            }
        return optimizer
        
    def log_sample_images(self, mode='train'):
        indices = getattr(self, f'smpl_indices_{mode}', None)
        sample_fake_imgs = self(getattr(self, f'smpl_latent_{mode}'), indices=indices)
        stack_imgs = torch.stack([getattr(self, f'smpl_target_{mode}'), sample_fake_imgs], dim=0)
        images = stack_imgs.permute(1,0,2,3,4).reshape(self.n_sample*2, *sample_fake_imgs.shape[1:])
        grid_imgs = torchvision.utils.make_grid(images, normalize=True, range=(-1,1), nrow=4)
        self.logger.experiment.add_image(f'{mode}_imgs', grid_imgs, self.current_epoch)
        
    def log_interpolated_images(self):
        N_INTERPOLATION = 10
        with torch.no_grad():
            latents = self.decoder(self.smpl_latent_train, indices=self.smpl_indices_train, get_latent=True)
        
            fake_img_list = []
            for idx in range(0, latents.shape[0], 2):
                latent_e1 = latents[idx]
                latent_e2 = latents[idx+1]
                step = (latent_e2 - latent_e1) / N_INTERPOLATION
                latent_list = list(map(lambda i: latent_e1 + i * step, range(1, N_INTERPOLATION)))
                latent_list = [latent_e1] + latent_list + [latent_e2]
                latent_list = [x.unsqueeze(0) for x in latent_list]
                batch_latent = torch.cat(latent_list, axis=0)

                fake_imgs = self(batch_latent, indices=None)
                res = fake_imgs.shape[2]
                fake_imgs = fake_imgs.permute(0,3,2,1)
                fake_imgs = fake_imgs.reshape(res*batch_latent.shape[0], res, 3)
                fake_imgs = fake_imgs.permute(2,1,0)
                fake_img_list.append(fake_imgs)
                
            interpolated_gallery = torch.cat(fake_img_list, dim=1).cpu().numpy()
            interpolated_gallery = (interpolated_gallery * 127.5 + 127.5).astype(np.uint8)
        self.logger.experiment.add_image('interpolated_img', interpolated_gallery, self.current_epoch)        