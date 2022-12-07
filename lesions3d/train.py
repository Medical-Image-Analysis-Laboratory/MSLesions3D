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
wandb.login()
import pickle
import time
from datetime import datetime
import argparse
import json
from os.path import join as pjoin
from os.path import exists as pexists
from pytorch_lightning.callbacks import EarlyStopping
from monai.utils import set_determinism
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset_path', type=str, help="path to dataset used for training and validation",
                    default=r'../data/artificial_dataset')
parser.add_argument('-dn', '--dataset_name', type=str, help="name of dataset to use", default="#3k_64_n1-5_s6-14")
parser.add_argument('-su', '--subject', type=str, default=None,
                    help="if training has to be done on 1 subject, specify its id")  # Set default to None
parser.add_argument('-p', '--percentage', type=float, default=1., help="percentage of the whole dataset to train on")
parser.add_argument('--n_classes', type=int, default=1, help="number of classes in dataset")
parser.add_argument('-b', '--batch_size', type=int, default=8, help="training batch size")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="training learning rate")
parser.add_argument('-sr', '--scheduler', type=str, default="CosineAnnealingLR", help="learning rate scheduler")
parser.add_argument('-th', '--threshold', type=float, default=[0.1,0.2], nargs='+', help="training IoU threshold for box matching (cf Amemiya 2021)")
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
parser.add_argument('-a', '--augmentations', type=str, nargs='*', default=["flip", "rotate90d", "translate"])
parser.add_argument('-ld', '--logdir', type=str, default=r'../logs/artificial_dataset')
parser.add_argument('-c', '--cache', type=int, default=0, help="whether to cache the dataset or not")
parser.add_argument('-nw', '--num_workers', type=int, default=8, help="number of workers for the dataset")
parser.add_argument('-wm', '--width_mult', type=float, default=1., help="width multiplicator (MobileNet)")
parser.add_argument('-en', '--experiment_name', type=str, default="multiple_subjects_64",
                    help="experiment name for tensorboard logdir")
parser.add_argument('-wb', '--use_wandb', type=int, default=1, help="whether to use weights and biases as logging tool")
parser.add_argument('-me', '--max_epochs', type=int, default=None, help="maximum number of epochs")
parser.add_argument('-mi', '--max_iterations', type=int, default=4000, help="maximum number of iterations")
parser.add_argument('-cp', '--checkpoint', type=str, default=None, help="path to model to load if resuming training")
parser.add_argument('-v', '--verbose', type=int, default=0, help="dataset verbose")
parser.add_argument('-rs', '--seed', type=int, default=970205, help="random seed")
parser.add_argument('-es', '--early_stopping', type=int, default=1, help="whether to use early stopping or not")
parser.add_argument('-cm', '--compute_metric_every_n_epochs', type=int, default=1, help="compute the metric every n epochs")
parser.add_argument('-coms', '--comments', type=str, default="", help="optional comments on the present run")


# Get the hyperparameters
args = parser.parse_args()
try:
    wandb.finish()
except:
    print("WandB Not running. Initializing!")
# Pass them to wandb.init
wandb.init(config=args)
# Access all hyperparameter values through wandb.config
args = wandb.config

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
    args.update({'max_iterations':-1}, allow_val_change=True)


def tune_lr():
    n_classes = 1
    model = LSSD3D(n_classes=n_classes + 1, input_channels=1, lr=3e-4,
                   width_mult=0.4)
    model.init()

    dataset = ExampleDataset(n_classes=n_classes, percentage=1., cache=True)
    dataset.setup(stage="fit")
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()

    # now = datetime.now()
    # logname = f"log_{now.day}-{now.month}-{now.year}_{now.hour}h{now.minute}_example"

    logdir = r"C:\Users\Cristina\Desktop\MSLesions3D\tensorboard\example"
    logger = TensorBoardLogger(logdir, name="full_dataset_160_40", default_hp_metric=False)
    # wandb_logger = WandbLogger(project="test")
    trainer = pl.Trainer(gpus=1, max_epochs=20, fast_dev_run=False, logger=logger,
                         enable_progress_bar=True, auto_lr_find=True,
                         log_every_n_steps=1)

    lr_finder = trainer.tuner.lr_find(model, train_loader)
    print(lr_finder.results)
    fig = lr_finder.plot(suggest=True)
    fig.show()


def pickle_dataset(dataset, dataset_file):
    import pickle
    print("Started pickling to", dataset_file)
    with open(dataset_file, "wb") as file:
        pickle.dump(dataset, file)


def example():
    pl.seed_everything(args.seed)
    set_determinism(seed=args.seed)

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

    dataset = ExampleDataset(n_classes=args.n_classes, subject=args.subject, percentage=args.percentage,
                             cache=args.cache, num_workers=args.num_workers, objects="multiple", verbose=bool(args.verbose),
                             batch_size=args.batch_size, augmentations=augmentations, data_dir=args.dataset_path,
                             dataset_name=args.dataset_name)
    dataset.setup(stage="fit")
    input_size = tuple(dataset.train_dataset[0]["img"].shape)[1:]

    model = LSSD3D(n_classes=args.n_classes + 1, input_channels=1, lr=args.learning_rate, width_mult=args.width_mult,
                   scheduler=args.scheduler, batch_size=args.batch_size, comments=args.comments, input_size=input_size,
                   compute_metric_every_n_epochs=5, use_wandb=args.use_wandb, aspect_ratios=aspect_ratios,
                   scales=scales, alpha=args.alpha, threshold=args.threshold, min_object_size=args.min_object_size,
                   max_object_size=args.max_object_size, base_network_config=args.base_network_config,
                   boxes_per_location=args.boxes_per_location)
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

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epochs, max_steps=args.max_iterations,
                         logger=logger, enable_progress_bar=True, log_every_n_steps=1,
                         callbacks=callbacks,
                         resume_from_checkpoint=args.checkpoint,
                         enable_checkpointing=True)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)


def train_lesions():
    pl.seed_everything(970205)

    dataset_file = r"C:\Users\Cristina\Desktop\MSLesions3D\data\lesions\dataset_bs8.pickle"

    augmentations = [("flip", {"spatial_axis": (0, 1, 2)}),
                     ("rotate90", {'spatial_axes': (1, 2)}),
                     ("affine", {"mode": ('bilinear', 'nearest'), "rotate_range": (np.pi / 12, np.pi / 12, np.pi / 12),
                                 "scale_range": (0.1, 0.1, 0.1), "padding_mode": 'border'}),
                     ("shiftintensity", {"offsets": 0.1, "prob": 1.0}),
                     ("scaleintensity", {"factors": 0.1, "prob": 1.0}),
                     ]

    batch_size = 8
    n_classes = 1
    lr = 0.01

    comments = f"""
    All subjects, test with augmentations (but without zoom)
    {augmentations}. 
    Now testing with a much higher LR
    """

    model = LSSD3D(n_classes=n_classes + 1, input_channels=1, lr=lr, width_mult=0.4, scheduler="CosineAnnealingLR",
                   batch_size=batch_size,
                   comments=comments, compute_metric_every_n_epochs=5)
    model.init()
    # dataset = LesionsDataModule(subject=('CHUV_RIM_OK', '010'), cache=True)
    dataset = LesionsDataModule(percentage=1., batch_size=batch_size, cache=True, random_state=None,
                                augmentations=augmentations, num_workers=1)
    dataset.setup(stage="fit")
    # pickle_dataset(dataset, dataset_file)
    # for data in dataset.train_dataset.data: print(data["subject"])

    # print("Loading dataset...")
    # s = time.time()
    # dataset = pickle.load(open(dataset_file, "rb"))
    # print(f"Dataset finished loading after {int(time.time() - s)}s")

    train_loader = dataset.train_dataloader()
    test_loader = dataset.val_dataloader()

    logdir = r"C:\Users\Cristina\Desktop\MSLesions3D\tensorboard\lesions"
    logger = TensorBoardLogger(logdir, name="zebardi", default_hp_metric=False)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=750, logger=logger, enable_progress_bar=True,
                         log_every_n_steps=1)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    # train_lesions()
    example()
    pass
