from argparse import ArgumentParser
from typing import List
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule

from pytorch_ssim import SSIM, ssim
from lpips_pytorch import LPIPS
from utils import mixing_noise
from .models import Generator, Discriminator
from losses import nonsaturating_loss, path_regularize, logistic_loss, d_r1_loss



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class LitSystem(LightningModule):
    def __init__(self,
                 log_sample_every,
                 samples,
                 losses: str = 'mse',
                 lr: float = 0.2,
                 lr_scheduler: str = None,
                 batch_size: int = 32,
                 latent_dim: int = 512,
                 train_size: int = 3200,
                 norm_type: str = 'batch_norm'
        ):
        
        super(LitSystem, self).__init__()
        
        # medium tutorial: https://medium.com/pytorch/pytorch-lightning-metrics-35cb5ab31857
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2406
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4030#issuecomment-708274317
        # register hyparams & metric in the init. and update value later
        self.save_hyperparameters("train_size", "val_size", "losses", "lr", "batch_size", "latent_dim",
                                  "norm_type")
        
        self.latent_dim = latent_dim
        self.mixing_prob = 0.9
        self.g_reg_every = 4
        self.d_reg_every = 16
        self.r1 = 10
        self.path_batch_shrink = 2
        self.path_regularize = 2
        
        self.g = Generator(latent_dim, 0, 256)
        self.d = Discriminator(0, 256)
        self.losses = losses.split(',')
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        
        # loss
        self.ce_loss = torch.nn.CrossEntropyLoss()
        # self.ssim_loss = SSIM()
        # https://github.com/S-aiueo32/lpips-pytorch
        # not using official repo. because we can't use sub-DataParallel model in pytorh-lightning
        # self.percept = LPIPS(
        #     net_type='alex',
        #     version='0.1'
        # )
        # for param in self.percept.parameters():
        #     param.requires_grad = False
        
        # metric & log
        self.log_sample_every = log_sample_every
        
        self.mean_path_length = 0
        self.register_buffer("path_lengths", torch.tensor(0.0))
        
        # sample
        n_sample = samples['train']['latents'].shape[0]
        
        self.register_buffer("sample_z", torch.randn(n_sample, latent_dim))
        self.register_buffer("smpl_latent_train", samples['train']['latents'])
        self.register_buffer("smpl_target_train", samples['train']['targets'])
        self.register_buffer("smpl_latent_test", samples['test']['latents'])
        self.register_buffer("smpl_target_test", samples['test']['targets'])
        self.n_sample = self.smpl_latent_train.shape[0]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-2)
        parser.add_argument('--losses', type=str, default='mse', help='comma seperated str. e.g. mse,lpips,ssim')
        parser.add_argument('--lr_scheduler', type=str, choices=['None', 'ReduceLROnPlateau', 'MultiStepLR'])
        parser.add_argument('--log_sample_every', type=int, default=10)
        
        return parser
        
    def forward(self, latent, skip_mapping=False):
        # inference
        return self.g([latent], skip_mapping=skip_mapping)

    def training_step(self, batch, batch_idx, optimizer_idx):
        latent, target_img = batch
        _, nz = latent.shape
        b, c, h, w = target_img.shape
        target_img = target_img[:b//2, ...] ### 
        
        # interpolated latents
        latent_l, latent_r = latent.view(2, b//2, nz)
        alpha = torch.rand((b//2,1)).type_as(latent)
        interpolated_latents = \
            latent_l * alpha + latent_r * (torch.ones_like(alpha) - alpha)

        use_reg = False #  if self.current_epoch > 3 else False
        (opt_gm, opt_gs, opt_d) = self.optimizers()
        
        # train mapping network
        # TODO: add mixing noise(for style mixing)
        # noise = mixing_noise(b, self.latent_dim, self.mixing_prob, )
        # noise = [z.type_as(latent) for z in noise]
        # latent = self.g.mapping_network(noise)
        # ce_loss = self.ce_loss(latent, ???)
        # opt_gm.zero_grad()
        # self.maunal_backward(ce_loss, opt_gm)
        # opt_gm.step()
        
        # train D
        requires_grad(self.g, False)
        requires_grad(self.d, True)
        fake_imgs_e, _ = self.g([interpolated_latents], skip_mapping=True) # real latent
        fake_pred = self.d(fake_imgs_e)
        real_pred = self.d(target_img)
        d_loss = logistic_loss(real_pred, fake_pred)
        d_loss_val = d_loss.detach()
        opt_d.zero_grad()
        self.manual_backward(d_loss, opt_d)
        opt_d.step()
        
        d_reg = batch_idx % self.d_reg_every == 0
        if use_reg and d_reg:
            target_img.requires_grad = True
            real_pred = self.d(target_img)
            r1_loss = d_r1_loss(real_pred, target_img)
            r1_loss = self.r1 / 2 * r1_loss * self.d_reg_every + 0 * real_pred[0]
            r1_loss_val = r1_loss.item()
            opt_d.zero_grad()
            self.manual_backward(r1_loss, opt_d)
            opt_d.step()
            self.log('Metric/D-r1-reg', r1_loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # train G
        requires_grad(self.g, True)
        requires_grad(self.d, False)
        # real latent & interpolated latent
        # fake_imgs_e, _ = self.g([latent], skip_mapping=True)
        # reconstruction loss
        # fake_imgs_e = torch.sigmoid(fake_imgs_e)
        # ssim_loss = - self.ssim_loss((target_img + 1) / 2., fake_imgs_e)
        # ssim_val = - ssim_loss.detach()
        
        fake_imgs_i, _ = self.g([interpolated_latents], skip_mapping=True)
        fake_pred_i = self.d(fake_imgs_i)
        
        g_loss = nonsaturating_loss(fake_pred_i)
        g_loss_val = g_loss.detach()
        opt_gs.zero_grad()
        self.manual_backward(g_loss, opt_gs)
        opt_gs.step()
        
        g_reg = batch_idx % self.g_reg_every == 0
        if use_reg and g_reg:
            # TODO: PPL on which latents(real or interpolated)
            path_loss, self.mean_path_length, self.path_lengths = path_regularize(
                fake_imgs_i, interpolated_latents, self.mean_path_length
            )
            weighted_path_loss = self.path_regularize * self.g_reg_every * path_loss
            weighted_path_loss_val = weighted_path_loss.item()
            opt_gs.zero_grad()
            self.manual_backward(weighted_path_loss, opt_gs)
            opt_gs.step()
            self.log('Metric/G-PPL-reg', weighted_path_loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('Metric/G_GANloss', g_loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Metric/D-GANloss', d_loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def training_epoch_end(self, training_step_outputs):
        if self.current_epoch == 0 or \
            (self.current_epoch+1) % self.log_sample_every == 0:
            self.log_sample_images()
            self.log_sample_images(mode='test')
            self.log_interpolated_images()
    
        # self.logger.log_hyperparams(
        #     params=self.hparams,
        #     metrics={
        #         
        #     }
        # )
        
    def configure_optimizers(self):
        g_reg_ratio = self.g_reg_every / (self.g_reg_every + 1)
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)
        opt_gm = torch.optim.Adam(
            self.g.mapping_network.parameters(),
            lr=self.lr * g_reg_ratio,
            betas=(0, 0.99))
        opt_gs = torch.optim.Adam(
            self.g.synthesis_network.parameters(),
            lr=self.lr*g_reg_ratio,
            betas=(0, 0.99))
        opt_d = torch.optim.Adam(
            self.d.parameters(),
            lr=self.lr*d_reg_ratio,
            betas=(0, 0.99))
        
        return [opt_gm, opt_gs, opt_d]
        
    def log_sample_images(self, mode='train'):
        sample_fake_imgs = self(getattr(self, f'smpl_latent_{mode}'), skip_mapping=True)[0]
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
                fake_imgs = self(batch_latent, skip_mapping=True)[0]
                res = fake_imgs.shape[2]
                fake_imgs = fake_imgs.permute(0,3,2,1)
                fake_imgs = fake_imgs.reshape(res*batch_latent.shape[0], res, 3)
                fake_imgs = fake_imgs.permute(2,1,0)
                fake_img_list.append(fake_imgs)
                
        interpolated_gallery = torch.cat(fake_img_list, dim=1).cpu().numpy()
        interpolated_gallery = (interpolated_gallery * 127.5 + 127.5).astype(np.uint8)
        self.logger.experiment.add_image('interpolated_img', interpolated_gallery, self.current_epoch)        