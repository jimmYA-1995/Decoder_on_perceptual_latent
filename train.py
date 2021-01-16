from pathlib import Path
from argparse import ArgumentParser
from random import sample

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from models import LitSystem, CNNDecoder
from dataset import get_dataloaders


def main(args):
    data_split = [args.train_size, args.val_size, args.test_size]
    
    (train_loader, val_loader, test_loader), train_latents = \
        get_dataloaders(args.root_dir, args.latent_path, args.target_dir, data_split,
                        args.num_workers, args.latent_dim, args.bs_per_gpu)
    
    samples = {}
    indices, _, target_img = next(iter(train_loader))
    samples['train'] = {
        'latents': indices[:args.num_sample],
        'targets': target_img[:args.num_sample]
    }
    latent, target_img = next(iter(test_loader))
    samples['test'] = {
        'latents': latent[:args.num_sample],
        'targets': target_img[:args.num_sample]
    }
    
    model = CNNDecoder(train_latents, args.norm_type)
    train_system = LitSystem(model,
                             args.log_sample_every,
                             # args.bs_per_gpu,
                             # args.train_size,
                             samples,
                             lr=args.lr,
                             lr_scheduler=args.lr_scheduler,
                             batch_size=(args.n_gpu*args.bs_per_gpu),
                             norm_type=args.norm_type,
                             latent_dim=args.latent_dim,
                             train_size=args.train_size,
                             val_size=args.val_size)

    tb_logger = pl_loggers.TensorBoardLogger(
        'runs', name=args.log_name, version=args.version, default_hp_metric=False)
    
    # maybe load from checkpoints
    kwargs = dict(logger=tb_logger, distributed_backend='ddp')
    if not args.resume_from_checkpoint:
        kwargs['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    trainer = Trainer.from_argparse_args(args, **kwargs)
    trainer.fit(train_system, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser()

    # general
    parser.add_argument('--num_sample', type=int, default=32)
    parser.add_argument('--log_name', type=str, default='default')
    parser.add_argument('--version', type=str, default=None)

    # data
    parser.add_argument('--root_dir', type=str, default='~/data/FFHQ')
    parser.add_argument('--latent_path', type=str, default='feat_PCA_L5_1024')
    parser.add_argument('--target_dir', type=str, \
                        default='images256x256', help="path to target(image) directory")
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--train_size', type=int, default=3200)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--test_size', type=int, default=1000)
    
    # model
    parser.add_argument('--bs_per_gpu', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--norm_type', type=str, default='batch_norm')
    
    parser = LitSystem.add_model_specific_args(parser)
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
