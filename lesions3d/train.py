# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:21:07 2022

@author: Maxence Wynen
"""

from datasets import *
from ssd3d import *
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl
import wandb
import argparse
import json
from os.path import join as pjoin
from os.path import exists as pexists
from pytorch_lightning.callbacks import EarlyStopping
from monai.utils import set_determinism
import random
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-wn', '--wandb_name', type=str, help="wandb run name", default=None)
parser.add_argument('-wp', '--wandb_project_name', type=str, help="wandb project name", default="PRL only")
parser.add_argument('-d', '--dataset_path', type=str, help="path to dataset used for training and validation",
                    default=r'../data/artificial_dataset')
parser.add_argument('-dn', '--dataset_name', type=str, help="name of dataset to use", default="#3k_64_n1-5_s6-14")
parser.add_argument('-seqs', '--sequences', type=str, nargs='+', help="sequences to use for training",
                    default=('FLAIR', 'acq-phase_T2star'))
parser.add_argument('-mf', '--metadata_file', type=str, help="metadata fileto use for training", default=None)
parser.add_argument('-sf', '--seg_filename', type=str, help="filename for the segmentations", default="seg")
parser.add_argument('-sm', '--seg_mode', type=str, help="segmentation mode ('instances', 'classes', 'binary')",
                    default="classes")
parser.add_argument('-st', '--seg_thresholds', type=int, help="segmentation thresholds (in case of seg_mode being "
                                                              "'instances')", default=None, nargs='*')
parser.add_argument('-igid', '--ignored_ids_file', type=str, help="file containing the ids to ignore", default=None)

parser.add_argument('--n_classes', type=int, default=1, help="number of classes in dataset")
parser.add_argument('-su', '--subject', type=str, default=None,
                    help="if training has to be done on 1 subject, specify its id")  # Set default to None
parser.add_argument('-p', '--percentage', type=float, default=1., help="percentage of the whole dataset to train on")
parser.add_argument('-imsi', '--image_size', type=int, default=None, nargs='+',
                    help="size of the images to use for training")
parser.add_argument('-psi', '--patch_size', type=int, default=None, nargs='+',
                    help="size of the cropped patches to use for training")

parser.add_argument('-b', '--batch_size', type=int, default=4, help="training batch size")
parser.add_argument('-ns', '--num_samples', type=int, default=8, help="how many samples to retrieve from 1 subject")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="training learning rate")
parser.add_argument('-sr', '--scheduler', type=str, default="CosineAnnealingLR", help="learning rate scheduler")
# parser.add_argument('-tmax', '--tmax', type=int, help="T_max argument for CosineAnnealingLR scheduler", default=40)
parser.add_argument('-ss', '--step_size', type=int, help="step_size argument for StepLR scheduler", default=1000)
parser.add_argument('-g', '--gamma', type=float, help="gamma argument for StepLR scheduler", default=0.5)

parser.add_argument('-a', '--augmentations', type=str, nargs='*', default=["flip", "rotate90d", "translate"])
parser.add_argument('-ld', '--logdir', type=str, default=r'../logs/artificial_dataset')
parser.add_argument('-cl', '--classification_loss', type=str, default=r'crossentropy', help="classification loss",
                    choices=['focal', 'crossentropy'],)
parser.add_argument('-c', '--cache', type=int, default=0, help="whether to cache the dataset or not")
parser.add_argument('-nw', '--num_workers', type=int, default=8, help="number of workers for the dataset")
parser.add_argument('-en', '--experiment_name', type=str, default="multiple_subjects_64",
                    help="experiment name for tensorboard logdir")
parser.add_argument('-wb', '--use_wandb', type=int, default=1, help="whether to use weights and biases as logging tool")
parser.add_argument('-me', '--max_epochs', type=int, default=None, help="maximum number of epochs")
parser.add_argument('-mi', '--max_iterations', type=int, default=4000, help="maximum number of iterations")
parser.add_argument('-cp', '--checkpoint', type=str, default=None, help="path to model to load if resuming training")
parser.add_argument('-es', '--early_stopping', type=int, default=1, help="whether to use early stopping or not")

parser.add_argument('-th', '--threshold', type=float, default=[0.1,0.2], nargs='+', help="training IoU threshold for box matching (cf Amemiya 2021)")
parser.add_argument('-wm', '--width_mult', type=float, default=1., help="width multiplicator (MobileNet)")
parser.add_argument('-pl', '--prediction_layers', type=str, default="3 5 7", help="feature maps on which to do the prediction convolutions.")
parser.add_argument('-cfg', '--base_network_config', type=str, default="mobilenet", help="base network configuration")
parser.add_argument('-sc', '--scales', type=json.loads, default="{}", help="Object scales per layer")
parser.add_argument('-bpl', '--boxes_per_location', type=int, default=2, help="Number of anchors per location in the feature maps")
parser.add_argument('-minos', '--min_object_size', type=int, default=6,
                    help="Minimum size for an object (for computation of scales). Not taken into account if scales argument is set.")
parser.add_argument('-maxos', '--max_object_size', type=int, default=14,
                    help="Minimum size for an object (for computation of scales). Not taken into account if scales argument is set.")
parser.add_argument('--alpha', type=int, default=1.,
                    help="alpha parameter for the multibox loss (= confidence loss + alpha * localization loss)")


parser.add_argument('-v', '--verbose', type=int, default=0, help="dataset verbose")
parser.add_argument('-rs', '--seed', type=int, default=1, help="random seed")
parser.add_argument('-cm', '--compute_metric_every_n_epochs', type=int, default=5, help="compute the metric every n epochs")

parser.add_argument('-coms', '--comments', type=str, default="", help="optional comments on the present run")


# Get the hyperparameters
args = parser.parse_args()
wandb.login()
try:
    wandb.finish()
except:
    print("WandB Not running. Initializing!")
# Pass them to wandb.init
wandb.init(config=args, project=args.project_name)
# Access all hyperparameter values through wandb.config
args = wandb.config
if args.wandb_name:
    wandb.run.name = args.wandb_name

try:
    layers = [int(x) for x in args.prediction_layers.split()]
except ValueError:
    print("Layers argument must be a sequence of integers separated by a space ' '")
    print("Run this script help to know more (--help)")
    exit()

aspect_ratios = {l: [1.] for l in layers}
scales = {int(k): v for k, v in args.scales.items()}
print(args)
print("Aspect ratios: ", aspect_ratios)
print("Scales: ", scales)
if args.max_epochs:
    args.update({'max_iterations': -1}, allow_val_change=True)
if args.seg_thresholds:
    st = [(int(args.seg_thresholds[i]), int(args.seg_thresholds[i+1]) if i + 1 < len(args.seg_thresholds) else float('inf')) \
            for i in range(0, len(args.seg_thresholds), 2)]
    args.update({'seg_thresholds': st}, allow_val_change=True)
    print(f"\n\nSeg_thresholds changed to: {args.seg_thresholds}\n\n")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
set_determinism(seed=args.seed)
pl.seed_everything(args.seed)

def example():
    augmentations = [("flip", {"spatial_axis": (0, 1, 2), "prob": .5}),
                     ("rotate90", {'spatial_axes': (1, 2), "prob": .5}),
                     ("rotate90", {'spatial_axes': (0, 1), "prob": .5}),
                     ("rotate90", {'spatial_axes': (0, 2), "prob": .5}),
                     ("translate", {"mode": ('bilinear', 'nearest'),
                                 "translate_range": (-3, 3),
                                 "prob": .7}),
                     # ("scale", {"mode": ('bilinear', 'nearest'),
                     #             "scale_range": (0.15, 0.15, 0.15), "padding_mode": 'reflection',
                     #             "prob": .7}),
                     ]

    augmentations = [(n.replace("translate", "affine").replace("scale","affine"), i)
                     for n, i in augmentations if n in args.augmentations]

    dataset = ExampleDataset(n_classes=args.n_classes,
                             subject=args.subject,
                             percentage=args.percentage,
                             cache=args.cache,
                             num_workers=args.num_workers,
                             objects="multiple",
                             verbose=bool(args.verbose),
                             batch_size=args.batch_size,
                             num_samples=args.num_samples,
                             augmentations=augmentations,
                             data_dir=args.dataset_path,
                             dataset_name=args.dataset_name,
                             seg_filename=args.seg_filename,
                             segmentation_mode=args.seg_mode,
                             seg_thresholds=args.seg_thresholds,
                             image_size=args.image_size,
                             patch_size=args.patch_size,)
    dataset.setup(stage="fit")
    dummy_input = dataset.train_dataset[0]
    dummy_input = dummy_input if type(dummy_input) == dict else dummy_input[0]
    input_size = tuple(dummy_input["img"].shape)[1:]

    model = LSSD3D(n_classes=args.n_classes + 1,
                   input_channels=1,
                   lr=args.learning_rate,
                   width_mult=args.width_mult,
                   scheduler=args.scheduler,
                   batch_size=args.batch_size*args.num_samples,
                   comments=args.comments,
                   input_size=input_size,
                   compute_metric_every_n_epochs=args.compute_metric_every_n_epochs,
                   use_wandb=args.use_wandb,
                   aspect_ratios=aspect_ratios,
                   scales=scales,
                   alpha=args.alpha,
                   threshold=args.threshold,
                   min_object_size=args.min_object_size,
                   max_object_size=args.max_object_size,
                   base_network_config=args.base_network_config,
                   boxes_per_location=args.boxes_per_location,
                   classification_loss=args.classification_loss,
                   t_max=args.max_epochs,
                   step_size=args.step_size,
                   gamma=args.gamma)
    model.init()

    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()

    logdir = args.logdir
    if not pexists(pjoin(logdir, args.experiment_name)):
        os.makedirs(pjoin(logdir, args.experiment_name))
    tb_logger = TensorBoardLogger(logdir, name=args.experiment_name, default_hp_metric=False)
    wandb_logger = WandbLogger(save_dir=args.logdir, project="MSLesions3D-lesions3d")
    logger = wandb_logger if args.use_wandb else tb_logger
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="avg_val_loss",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="checkpoint-{epoch:03d}-{avg_val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    callbacks = [checkpoint_callback]
    if args.early_stopping:
        print("Early stopping strategy")
        callbacks += [EarlyStopping('total_loss/validation', patience=5)]

    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=args.max_epochs,
                         max_steps=args.max_iterations,
                         logger=logger,
                         enable_progress_bar=True,
                         log_every_n_steps=1,
                         callbacks=callbacks,
                         resume_from_checkpoint=args.checkpoint,
                         enable_checkpointing=True)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)


def train_lesions():
    augmentations = [("flip", {"spatial_axis": (0, 1, 2), "prob": .5}),
                     ("rotate90", {'spatial_axes': (1, 2), "prob": .5}),
                     ("rotate90", {'spatial_axes': (0, 1), "prob": .5}),
                     ("rotate90", {'spatial_axes': (0, 2), "prob": .5}),
                     ("translate", {"mode": ('bilinear', 'nearest'),
                                 "translate_range": (-3, 3),
                                 "prob": .7}),
                     ("scale", {"mode": ('bilinear', 'nearest'),
                                 "scale_range": (0.15, 0.15, 0.15), "padding_mode": 'reflection',
                                 "prob": .7}),
                     ]

    augmentations = [(n.replace("translate", "affine").replace("scale","affine"), i)
                     for n, i in augmentations if n in args.augmentations]

    dataset = ObjectDetectionDataset(n_classes=args.n_classes,
                                     subject=args.subject,
                                     percentage=args.percentage,
                                     cache=args.cache,
                                     num_workers=args.num_workers,
                                     verbose=bool(args.verbose),
                                     batch_size=args.batch_size,
                                     num_samples=args.num_samples,
                                     augmentations=augmentations,
                                     data_dir=args.dataset_path,
                                     seg_filename=args.seg_filename,
                                     segmentation_mode=args.seg_mode,
                                     seg_thresholds=args.seg_thresholds,
                                     image_size=args.image_size,
                                     patch_size=args.patch_size,
                                     input_images=args.sequences,
                                     metadata_file=args.metadata_file,
                                     ignored_ids_file=args.ignored_ids_file,)
    dataset.setup(stage="fit")
    dummy_input = dataset.train_dataset[0]
    dummy_input = dummy_input if type(dummy_input) == dict else dummy_input[0]
    input_size = tuple(dummy_input["img"].shape)[1:]

    model = LSSD3D(n_classes=args.n_classes + 1,
                   input_channels=len(args.sequences),
                   lr=args.learning_rate,
                   width_mult=args.width_mult,
                   scheduler=args.scheduler,
                   batch_size=args.batch_size*args.num_samples,
                   comments=args.comments,
                   input_size=input_size,
                   compute_metric_every_n_epochs=args.compute_metric_every_n_epochs,
                   use_wandb=args.use_wandb,
                   aspect_ratios=aspect_ratios,
                   scales=scales,
                   alpha=args.alpha,
                   threshold=args.threshold,
                   min_object_size=args.min_object_size,
                   max_object_size=args.max_object_size,
                   base_network_config=args.base_network_config,
                   boxes_per_location=args.boxes_per_location,
                   classification_loss=args.classification_loss,
                   t_max=args.max_epochs,
                   step_size=args.step_size,
                   gamma=args.gamma)
    model.init()

    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    logdir = args.logdir
    if not pexists(pjoin(logdir, args.experiment_name)):
        os.makedirs(pjoin(logdir, args.experiment_name))
    tb_logger = TensorBoardLogger(logdir, name=args.experiment_name, default_hp_metric=False)
    wandb_logger = WandbLogger(save_dir=args.logdir, project="MSLesions3D-lesions3d")
    logger = wandb_logger if args.use_wandb else tb_logger
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="avg_val_loss",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="checkpoint-{epoch:03d}-{avg_val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    callbacks = [checkpoint_callback]
    if args.early_stopping:
        print("Early stopping strategy")
        callbacks += [EarlyStopping('total_loss/validation', patience=5)]

    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=args.max_epochs,
                         max_steps=args.max_iterations,
                         logger=logger,
                         enable_progress_bar=True,
                         log_every_n_steps=1,
                         callbacks=callbacks,
                         resume_from_checkpoint=args.checkpoint,
                         enable_checkpointing=True)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train_lesions()
    #example()
    pass
