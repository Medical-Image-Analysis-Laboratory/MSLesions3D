# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:21:07 2022

@author: Maxence Wynen
"""

from datasets import *
from ssd3d import *
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl
# import wandb
# wandb.login()
import pickle
import time
from datetime import datetime
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', type=str, help="path to dataset used for training and validation",
                    default=r'/home/wynen/MSLesions3D/data/artificial_dataset/multiple_objects/one_class/images')
parser.add_argument('-su', '--subject', type=str, default='0000', help="if training has to be done on 1 subject, specify its id") # Set default to None
parser.add_argument('-p', '--percentage', type=float, default=1., help="percentage of the whole dataset to train on")
parser.add_argument('--n_classes', type=int, default=1, help="number of classes in dataset")
parser.add_argument('-b', '--batch_size', type=int, default=8, help="training batch size")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help="training learning rate")
parser.add_argument('-sr', '--scheduler', type=str, default="CosineAnnealingLR", help="learning rate scheduler")
parser.add_argument('-l', '--layers', type=int, nargs='+', default=[3, 5, 7], help="layers to include in the network")
parser.add_argument('-sc', '--scales', type=json.loads, default="{\"1\": 0.05, \"3\": 0.075, \"5\": 0.1, \"7\": 0.125}",
                    help="layers to include in the network")
parser.add_argument('--alpha', type=int, default=[3, 5, 7],
                    help="alpha parameter for the multibox loss (= confidence loss + alpha * localization loss)")
parser.add_argument('-a', '--augmentations', type=bool, default=False)
parser.add_argument('-ld', '--logdir', type=str, default=r'/home/wynen/MSLesions3D/artificial_dataset')
parser.add_argument('-c', '--cache', type=bool, default=False, help="whether to cache the dataset or not")
parser.add_argument('-nw', '--num_workers', type=int, default=8, help="number of workers for the dataset")
parser.add_argument('-wm', '--width_mult', type=float, default=0.4, help="width multiplicator (MobileNet)")

args = parser.parse_args()
ARS = {l: [1.] for l in args.layers}
SC = {int(k):v for k,v in args.scales.items()}
print(args)
print("Aspect ratios: ", ARS)
print("Scales: ", SC)


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
    pl.seed_everything(970205)

    n_classes = args.n_classes
    batch_size = args.batch_size
    lr = args.learning_rate
    scheduler = args.scheduler
    augmentations = [("flip", {"spatial_axis": (0, 1, 2), "prob": .5}),
                     ("rotate90", {'spatial_axes': (0, 1, 2), "prob": .5}),
                     ("affine", {"mode": ('bilinear', 'nearest'),
                                 "scale_range": (0.15, 0.15, 0.15), "padding_mode": 'reflection',
                                 "translate_range": (-15, 15),
                                 "shear_range": (-.1, .1),
                                 "prob": .7}),
                     ] if args.augmentations else None



    ASPECT_RATIOS = ARS
    SCALES = SC
    alpha = 1.
    comments = f"""
    
    Let's try with with ASPECT_RATIO = {ASPECT_RATIOS} and SCALES =  {SCALES}
    
    Back to alpha = {alpha}
    
    Augmentations = {augmentations}
    
    """

    model = LSSD3D(n_classes=args.n_classes + 1, input_channels=1, lr=args.learning_rate, width_mult=args.width_mult,
                   scheduler=args.scheduler, batch_size=args.batch_size, comments=comments,
                   compute_metric_every_n_epochs=5, use_wandb=False, ASPECT_RATIOS=ASPECT_RATIOS, SCALES=SCALES)
    model.init()

    # dataset = ExampleDataset(n_classes = n_classes, subject = "0420", percentage = -1, cache=True, num_workers=1, objects="multiple", batch_size=batch_size)
    dataset = ExampleDataset(n_classes=args.n_classes, subject = args.subject, percentage=args.percentage, cache=args.cache,
                             num_workers=args.num_workers, objects="multiple", batch_size=args.batch_size,
                             augmentations=augmentations)
    dataset.setup(stage="fit")

    # # pickle_dataset(dataset, dataset_file)
    # for data in dataset.train_dataset.data: print(data["subject"])

    # import time
    # print("Loading dataset...")
    # s = time.time()
    # dataset = pickle.load(open(dataset_file, "rb"))
    # print(f"Dataset finished loading after {int(time.time() - s)}s")

    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()

    logdir = args.logdir
    logger = TensorBoardLogger(logdir, name="one_subject_64", default_hp_metric=False)
    # wandb_logger = WandbLogger(project="WhiteBoxes")
    trainer = pl.Trainer(gpus=1, max_epochs=100, logger=logger, enable_progress_bar=True, log_every_n_steps=1)
    # trainer = pl.Trainer(gpus=1, max_epochs=600, fast_dev_run=False, logger=logger,  enable_progress_bar=True, log_every_n_steps=1,
    #                      resume_from_checkpoint = r"C:\Users\Cristina\Desktop\MSLesions3D\tensorboard\example\full_dataset_400_100\version_21\checkpoints\epoch=63-step=3200.ckpt")

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
    trainer = pl.Trainer(gpus=1, max_epochs=750, logger=logger, enable_progress_bar=True, log_every_n_steps=1)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    # train_lesions()
    example()
    pass
