import torch
# import json
import os
# from os.path import exists as pexists
from os.path import join as pjoin
from utils import BoundingBoxesGeneratord, Printer, ShowImage
from random import randint
from monai.data import Dataset, CacheDataset, DataLoader
from datasets import ExampleDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    CropForegroundd,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Lambdad,
    RandRotate90d,
    RandGridDistortiond,
    RandFlipd,
    RandZoomd,
    Zoomd,
    SaveImaged,
    RandAffined,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    Resized,
    SaveImageD
)
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# out_dir = r"C:\Users\Cristina\Desktop\MSLesions3D\data\deleteme"
# data_dir=r"C:\Users\Cristina\Desktop\2DFA\data\raw"
#
# input_images=("FLAIR",)
# segmentation="labeled_lesions"
# classes=('lesion',),
# registration='T2star'
# skullstripped=True
# subjects = [('CHUV_RIM_OK', "010")]
#
#
# def _get_data_dir(center):
#     '''Returns path to the right BIDS directory'''
#     dd = pjoin(data_dir, center)
#     if registration is not None:
#         dd = pjoin(dd, 'derivatives', 'registrations', f'registrations_to_{registration}')
#     return dd
#
# def _get_sequence(center, subject, img_name):
#     '''Returns path to the image'''
#     # if the image is an MRI sequence
#     if img_name in ['FLAIR', 'acq-phase_T2star', 'acq-mag_T2star']:
#         if not skullstripped:
#             path = pjoin(_get_data_dir(center), f"sub-{subject}", "ses-01", "anat",
#                          f"sub-{subject}_ses-01_{img_name}.nii.gz")
#         else:
#             path = pjoin(_get_data_dir(center), "derivatives", "skullstripped", f"sub-{subject}",
#                          "ses-01", f"sub-{subject}_ses-01_{img_name}.nii.gz")
#     # Segmentations
#     else:
#         path = pjoin(_get_data_dir(center), "derivatives", "lesionmasks", f"sub-{subject}",
#                      "ses-01", f"sub-{subject}_ses-01_{img_name}.nii.gz")
#     return path
# zoom = 1.5
# def save_image(to_test=None):
#
#     if not to_test:
#         transforms = Compose([LoadImaged(keys=["img", "seg"]),
#                               AddChanneld(keys=["img", "seg"]),
#                               Orientationd(keys=["img", "seg"], axcodes="LPI"),
#                               CropForegroundd(keys=["img", "seg"], source_key= "img", mode= "constant", margin= 5),
#                               ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size= (250, 300, 300), mode ="constant"),
#                               SaveImaged(keys=["img"], output_dir=out_dir, output_postfix=f"original"),
#                               SaveImaged(keys=["seg"], output_dir=out_dir, output_postfix=f"original"),
#                               ToTensord(keys="img")
#                               ])
#     else:
#         transforms = Compose([LoadImaged(keys=["img", "seg"]),
#                               AddChanneld(keys=["img","seg"]),
#                               Orientationd(keys=["img","seg"], axcodes="LPI"),
#                               CropForegroundd(keys=["img","seg"], source_key= "img", mode= "constant", margin= 5),
#                               to_test,
#                               # ResizeWithPadOrCropd(keys=["img","seg"], spatial_size= (250, 300, 300), mode ="constant"),
#                               SaveImaged(keys=["img"], output_dir=out_dir, output_postfix=f"transformed"),
#                               SaveImaged(keys=["seg"], output_dir=out_dir, output_postfix=f"transformed"),
#                               ToTensord(keys="img")
#                               ])
#
#
#     data = [{'img': _get_sequence(c, s, input_images[0]),
#                'seg': _get_sequence(c, s, segmentation),
#                'center': c, 'subject': s} for c, s in subjects]
#
#
#     dataset = Dataset(data=data, transform=transforms)
#
#     return dataset[0]


# to_test = RandFlipd(keys=["img", "seg"], prob=0.5 ,spatial_axis=(0 ,1 ,2))
# to_test = RandAffined(keys=["img", "seg"], prob=1.0, mode=('bilinear', 'nearest'), rotate_range=(np.pi /12, np.pi /12, np.pi /12),  scale_range=(0.1, 0.1, 0.1), padding_mode='border')
# to_test = Zoomd(keys=["img","seg"], zoom=zoom, keep_size =True, padding_mode="constant", mode=('bilinear', 'nearest')),
# to_test = Compose([RandCropByPosNegLabeld(keys=["img","seg"], label_key="seg", spatial_size=(150,175,175)),
#                    Resized(keys=["img","seg"], spatial_size =(250,300,300), mode = ('bilinear', 'nearest'))
#                    ])

# orig =  save_image()
# trans = save_image(to_test)
n_classes=1
batch_size = 8
augmentations = [("flip", {"spatial_axis": (0, 1, 2), "prob":.5}),
                 ("rotate90", {'spatial_axes': (1, 2), "prob":.5}),
                 ("affine", {"mode": ('bilinear', 'nearest'),
                             "scale_range": (0.15, 0.15, 0.15), "padding_mode": 'reflection',
                             "translate_range":(-15,15),
                             "shear_range":(-.1,.1),
                             "prob":.7}),
                 ("shiftintensity", {"offsets": (-0.2,0.2), "prob": 1.0}),
                 # ("scaleintensity", {"factors": 0.2, "prob": 1.0}),
                 ]

dataset = ExampleDataset(n_classes = n_classes, percentage = 1., cache=False, num_workers=8, objects="multiple", batch_size=batch_size,
                             augmentations=augmentations, show=True)
dataset.setup(stage="fit")

inp = input("Continue?")
while inp != "stop":
    img = dataset.train_dataset[0]
    SaveImageD(keys=("img", "seg"))(img)
    inp = input("Press Enter to continue...")