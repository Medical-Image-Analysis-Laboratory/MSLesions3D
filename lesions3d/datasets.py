# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:56:29 2022

@author: Maxence Wynen
"""
import json
import warnings

import torch
import yaml
import os
from os.path import exists as pexists
from os.path import join as pjoin
from utils import *
from random import randint
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    CropForegroundd,
    FgBgToIndicesd,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    NormalizeIntensityd,
    Lambdad,
    RandRotate90d,
    RandGridDistortiond,
    RandFlipd,
    RandZoomd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAffined,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    Spacingd,
    CastToTyped,
    RandGaussianSmoothd,
    RandGaussianNoised,
)
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import logging

# from warnings import filterwarnings
# filterwarnings('ignore')
EXCLUDED_SUBJECTS = [("BASEL_INSIDER_OK", "085")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    This describes how to combine these tensors of different sizes. We use lists.

    :param batch: an iterable of N sets from __getitem__()
    :return: a dict of images, lists of varying-size tensors of bounding boxes, labels
    """

    images = list()
    boxes = list()
    labels = list()
    subjects = list()
    img_meta_dicts = list()
    seg_meta_dicts = list()
    seg_transforms = list()
    img_transforms = list()

    def add_data_to_lists(b):
        if "boxes" not in b.keys():
            batch_imgs, [batch_boxes, batch_labels] = b["img"], b['seg']
        else:
            batch_imgs = b["img"]
            batch_boxes, batch_labels = b["boxes"], b["labels"]
            if len(b["labels"]) == 0:  # filter out empty images
                return

        images.append(batch_imgs)
        boxes.append(batch_boxes)
        labels.append(batch_labels)
        subjects.append(b["subject"])
        img_meta_dicts.append(b["img_meta_dict"])
        seg_meta_dicts.append(b["seg_meta_dict"])
        seg_transforms.append(b["seg_transforms"])
        img_transforms.append(b["img_transforms"])

    for b in batch:
        if type(b) == dict:
            add_data_to_lists(b)
        else:  # type(b) == list
            for b2 in b:
                add_data_to_lists(b2)

    images = torch.stack(images, dim=0)

    batch_ = {"img": images,
              "seg": [boxes, labels],
              "boxes": boxes,
              "labels": labels,
              "subject": subjects,
              "img_meta_dict": img_meta_dicts,
              "seg_meta_dict": seg_meta_dicts,
              "img_transforms": img_transforms,
              "seg_transforms": seg_transforms}

    return batch_


def collate_fn_predict(batch):
    """
        Collate fn for prediction (not taking the ground truth into account)

        :param batch: an iterable of N sets from __getitem__()
        :return: a dict of images, lists of varying-size tensors of bounding boxes, labels
    """

    images = list()
    subjects = list()
    img_meta_dicts = list()
    img_transforms = list()

    for b in batch:
        if not "boxes" in b.keys():
            batch_imgs = b["img"]
        else:
            batch_imgs = b["img"]
        images.append(batch_imgs)
        subjects.append(b["subject"])
        img_meta_dicts.append(b["img_meta_dict"])
        img_transforms.append(b["img_transforms"])

    images = torch.stack(images, dim=0)

    batch_ = {"img": images,
              "subject": subjects,
              "img_meta_dict": img_meta_dicts,
              "img_transforms": img_transforms}

    return batch_


def get_transform_from_name(name, **kwargs):
    TRANSFORMS = {
        "add_channel": (AddChanneld, ["img", "seg"]),
        "add_channel_img": (AddChanneld, ["img"]),
        "add_channel_seg": (AddChanneld, ["seg"]),
        "affine": (RandAffined, ["img", "seg"]),
        "bounding_boxes_generator": (BoundingBoxesGeneratord, ["seg"]),
        "cast_to_type_img": (CastToTyped, ["img"]),
        "cast_to_type_seg": (CastToTyped, ["seg"]),
        "correct_label_affine": (CorrectLabelAffined, ["img", "seg"]),
        "crop_by_pos_neg": (RandCropByPosNegLabeld, ["img", "seg"]),
        "crop_foreground": (CropForegroundd, ["img", "seg"]),
        "fgbg_to_indices": (FgBgToIndicesd, ["seg"]),
        "flip": (RandFlipd, ["img", "seg"]),  # spatial_axis=(0 ,1 ,2)
        "gaussian_noise": (RandGaussianNoised, ["img"]),
        "gaussian_smooth": (RandGaussianSmoothd, ["img"]),
        "griddistortion": (RandGridDistortiond, ["img", "seg"]),  # {"padding_mode": "zeros"}
        "load_image": (LoadImaged, ["img", "seg"]),
        "normalizeintensity": (NormalizeIntensityd, ["img"]),
        "orientation": (Orientationd, ["img", "seg"]),
        "resize_with_pad_or_crop": (ResizeWithPadOrCropd, ['img', 'seg']),
        "rotate90": (RandRotate90d, ["img", "seg"]),  # kwargs = {'spatial_axes' = (1,2)}
        "scale_intensity": (ScaleIntensityd, ['img']),
        "scaleintensity": (RandScaleIntensityd, ["img"]),
        "shiftintensity": (RandShiftIntensityd, ["img"]),
        "spacing": (Spacingd, ["img", "seg"]),
        "spatial_crop_samples": (RandSpatialCropSamplesd, ["img", "seg"]),
        "to_tensor": (ToTensord, ["img"]),
        "zoom": (RandZoomd, ["img", "seg"]),

    }

    transform, keys = TRANSFORMS[name]
    return transform(keys=keys, **kwargs)

CastToTyped(keys=["image"], dtype=(torch.float32)),


class LesionsDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir=r"C:\Users\Cristina\Desktop\2DFA\data\raw",
                 centers=('CHUV_RIM_OK', 'BASEL_INSIDER_OK'),
                 fold=None,
                 input_images=("FLAIR",),
                 segmentation="labeled_lesions",
                 classes=('lesion',),
                 registration='T2star',
                 skullstripped=True,
                 augmentations=None,
                 subject=None,
                 batch_size=8,
                 percentage=1.,
                 verbose=False,
                 show=False,
                 num_workers=int(os.cpu_count() / 2),
                 random_state=970205,
                 cache=False):

        super().__init__()
        self.data_dir = data_dir
        self.centers = centers
        self.registration = registration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.random_state = random_state
        self.skullstripped = skullstripped
        self.input_images = input_images
        if len(input_images) != 1:
            raise NotImplementedError("Only supports one sequence at a time.")
        self.segmentation = segmentation
        self.percentage = percentage,
        self.cache = cache
        self.verbose = verbose
        self.show = show
        self.augmentations = augmentations
        self.subject = subject
        self.classes = classes
        self.n_classes = len(classes)
        self.segmentation_mode = "instances" if "labeled" in self.segmentation else "classes"
        if self.segmentation_mode == "classes":
            self.thresholds = None
        elif self.n_classes == 1:
            self.thresholds = [(1, np.inf)]
        elif self.n_classes == 2:
            self.thresholds = [(1000, 2000), (2000, np.inf)]

        self.data_dirs = []
        for c in centers:
            self.data_dirs.append(self._get_data_dir(c))

        self.subjects = {}
        self.subjects_list = []
        for c, dd in enumerate(self.data_dirs):
            csubjs = [s.replace("sub-", '') for s in os.listdir(dd) if 'sub-' in s]
            self.subjects[self.centers[c]] = csubjs
            self.subjects_list += [(self.centers[c], csubj) for csubj in csubjs]

        self.subjects_list = [x for x in self.subjects_list if x not in EXCLUDED_SUBJECTS]

        self.subjects_list = self.subjects_list[:int(percentage * len(self.subjects_list))] if percentage > 0 \
            else self.subjects_list

        self.train_transform = Compose(self.get_list_of_transforms("train"))
        self.test_transform = Compose(self.get_list_of_transforms("test"))

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers}  # , 'collate_fn': collate_fn}

    def get_list_of_transforms(self, mode):
        """
        Return list of transforms for both train and test datasets
        """
        base_list_start = [("load_image", {}),
                           ("add_channel", {}),
                           ("orientation", {"axcodes": "LPI"}),
                           ("spacing", {'pixdim': (1.0, 1.0, 1.0), "mode": ("bilinear", "nearest")}),
                           ("crop_foreground", {"source_key": "img", "mode": "constant", "margin": 5}),
                           ("normalizeintensity", {"nonzero": True}),
                           ]
        base_list_end = [("resize_with_pad_or_crop", {"spatial_size": (250, 300, 300), "mode": "replicate"}),
                         ("bounding_boxes_generator", {"segmentation_mode": self.segmentation_mode,
                                                       "n_classes": self.n_classes, "thresholds": self.thresholds}),
                         ("to_tensor", {}),
                         ]

        list_of_transforms = list()

        # First add the images, reorient them and make sure the channel is first
        for t_name, kwargs in base_list_start:
            if self.verbose: list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
            list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
            if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))

        # Perform of augmentations if specified and if training mode
        if self.augmentations is not None and mode == "train":
            for transform in self.augmentations:
                if type(transform) != tuple:
                    transform = (transform, {})
                t_name, kwargs = transform

                if self.verbose: list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
                list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
                if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))

        # Add the final transformations
        for t_name, kwargs in base_list_end:
            if self.verbose: list_of_transforms.append(Lambdad(keys=["seg"], func=Printer(t_name)))
            list_of_transforms.append(get_transform_from_name(t_name, **kwargs))

        return list_of_transforms

    def _get_data_dir(self, center):
        '''Returns path to the right BIDS directory'''
        dd = pjoin(self.data_dir, center)
        if self.registration is not None:
            dd = pjoin(dd, 'derivatives', 'registrations', f'registrations_to_{self.registration}')
        return dd

    def _get_sequence(self, center, subject, img_name):
        '''Returns path to the image'''
        # if the image is an MRI sequence
        if img_name in ['FLAIR', 'acq-phase_T2star', 'acq-mag_T2star']:
            if not self.skullstripped:
                path = pjoin(self._get_data_dir(center), f"sub-{subject}", "ses-01", "anat",
                             f"sub-{subject}_ses-01_{img_name}.nii.gz")
            else:
                path = pjoin(self._get_data_dir(center), "derivatives", "skullstripped", f"sub-{subject}",
                             "ses-01", f"sub-{subject}_ses-01_{img_name}.nii.gz")
        # Segmentations
        else:
            path = pjoin(self._get_data_dir(center), "derivatives", "lesionmasks", f"sub-{subject}",
                         "ses-01", f"sub-{subject}_ses-01_{img_name}.nii.gz")
        return path

    def setup(self, stage=None):
        # Only one subject, for debugging purposes
        if self.subject is not None:
            self.trainsubs, self.testsubs = [self.subject], [self.subject]

        # Picking a subject at random, for debugging purposes
        elif self.percentage == -1:
            random_subj = randint(0, len(self.subjects_list))
            print("Picked subject", self.subjects_list[random_subj])
            self.trainsubs, self.testsubs = [self.subjects_list[random_subj]], [self.subjects_list[random_subj]]

        # If not debugging, use all the dataset
        else:
            spl = train_test_split(self.subjects_list, train_size=0.8, test_size=0.2, random_state=self.random_state)
            self.trainsubs, self.testsubs = spl

        # If a fold was specified, split into 4 folds
        if self.fold is not None and stage != "all":
            kf = KFold(n_splits=4, shuffle=True, random_state=self.random_state)
            l = list(kf.split(self.trainsubs))

            train_indices = [train_index for train_index, _ in l]
            val_indices = [val_index for _, val_index in l]

            train_subs = self.trainsubs[train_indices[self.fold]]
            test_subs = self.testsubs[val_indices[self.fold]]
        # Otherwise use all the subjects
        else:
            train_subs = self.trainsubs
            test_subs = self.testsubs

        DS = CacheDataset if self.cache else Dataset

        data_train = [{'img': self._get_sequence(c, s, self.input_images[0]),
                       'seg': self._get_sequence(c, s, self.segmentation),
                       'center': c, 'subject': s} for c, s in train_subs]

        data_test = [{'img': self._get_sequence(c, s, self.input_images[0]),
                      'seg': self._get_sequence(c, s, self.segmentation),
                      'center': c, 'subject': s} for c, s in test_subs]

        data_all = [{'img': self._get_sequence(c, s, self.input_images[0]),
                     'seg': self._get_sequence(c, s, self.segmentation),
                     'center': c, 'subject': s} for c, s in self.subjects_list]

        if stage == "all":
            self.all_dataset = DS(data=data_all, transform=self.test_transform)

        if stage == "fit" or stage is None:
            self.train_dataset = DS(data=data_train, transform=self.train_transform)
            self.val_dataset = DS(data=data_test, transform=self.test_transform)

        if stage == 'predict' or stage is None:
            self.predict_train_dataset = DS(data=data_train, transform=self.test_transform)

        if stage == "test" or stage == 'predict' or stage is None:
            self.test_dataset = DS(data=data_test, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, collate_fn=collate_fn, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, collate_fn=collate_fn, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, collate_fn=collate_fn, **self.dl_dict)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, collate_fn=collate_fn_predict, **self.dl_dict)

    def predict_train_dataloader(self):
        return DataLoader(self.predict_train_dataset, shuffle=False, collate_fn=collate_fn_predict, **self.dl_dict)

    def all_dataloader(self):
        return DataLoader(self.all_dataset, shuffle=False, collate_fn=collate_fn, **self.dl_dict)


class ObjectDetectionDataset(pl.LightningDataModule):
    def __init__(self,
                 data_dir="/home/wynen/data/rimnet",
                 input_images=('FLAIR', 'acq-phase_T2star'),
                 metadata_file=None,
                 ignored_ids_file=None,
                 skullstripped=False,
                 n_classes=1,
                 percentage=1.,
                 augmentations=None,
                 batch_size=8,
                 num_workers=int(os.cpu_count() / 2),
                 verbose=False,
                 show=False,
                 random_state=970205,
                 cache=False,
                 subject=None,
                 seg_filename="bboxes_prl",
                 segmentation_mode="instances",
                 seg_thresholds=((1000, 2000), (2000, np.inf)),
                 image_size=None,
                 patch_size=None,
                 num_samples=1,
                 predict_subdir="images"):

        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_train_dataset = None
        self.predict_test_dataset = None
        self.predict_val_dataset = None
        self.predict_dataset = None

        self.data_dir = data_dir
        self.predict_subdir = predict_subdir
        self.input_images = input_images
        self.ignore_ids = []
        if ignored_ids_file is None:
            ignored_ids_file = os.path.join(self.data_dir, "ignored_ids.json")
        if not pexists(ignored_ids_file):
            print("No ignored_ids file found, ignoring no subjects")
        with open(ignored_ids_file, "r") as f:
            self.ignore_ids = json.load(f)["ignore"]

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.random_state = random_state
        self.cache = cache
        self.percentage = percentage
        self.verbose = verbose
        self.segmentation_mode = segmentation_mode
        self.seg_thresholds = seg_thresholds
        if self.segmentation_mode == "instances" and seg_thresholds is None:
            self.seg_thresholds = [(1, np.inf)]
            warnings.warn("No thresholds specified for segmentation. Using (1, np.inf) as default")
        self.n_classes = n_classes
        self.augmentations = augmentations
        self.show = show
        self.fold = None

        if metadata_file is None:
            metadata_file = pjoin(self.data_dir, "dataset.yaml")
        assert pexists(metadata_file) and metadata_file.endswith(
            ".yaml"), f"Metadata file {metadata_file} does not exist"
        with open(metadata_file) as f:
            self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
        inverse_modalities = {v: k for k, v in self.metadata["modalities"].items()}
        self.input_image_ids = [inverse_modalities[i].zfill(4) for i in self.input_images]

        self.all_trainsubs = list(set([sub.strip(".nii.gz")[:-5] for sub in os.listdir(pjoin(self.data_dir, "imagesTr")) \
                              if sub.strip(".nii.gz")[:-5] not in self.ignore_ids]))
        self.testsubs = list(set([sub.strip(".nii.gz")[:-5] for sub in os.listdir(pjoin(self.data_dir, "imagesTs")) \
                         if sub.strip(".nii.gz")[:-5] not in self.ignore_ids]))
        self.all_trainsubs = self.all_trainsubs[:int(percentage * len(self.all_trainsubs))] if percentage > 0 \
            else self.all_trainsubs
        self.test_subjects_list = self.testsubs[:int(percentage * len(self.testsubs))] if percentage > 0 \
            else self.testsubs
        self.trainsubs = None
        self.valsubs = None
        self.predictsubs = None
        self.subject = subject
        self.seg_filename = seg_filename

        assert image_size or patch_size, "Either image_size or patch_size must be specified"
        self.image_size = image_size
        self.patch_size = patch_size if patch_size is not None else image_size

        self.train_transform_list = self.get_list_of_transforms("train")
        self.test_transform_list = self.get_list_of_transforms("test")
        self.predict_transform_list = self.get_list_of_transforms("predict")

        self.train_transform = Compose(self.train_transform_list)
        self.test_transform = Compose(self.test_transform_list)
        self.predict_transform = Compose(self.predict_transform_list)

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers}

    def get_list_of_transforms(self, mode):
        """
        Return list of transforms for train, test or predict datasets
        """
        base_list_start = [("load_image", {}),
                           ("add_channel_seg", {}),
                           ("orientation", {"axcodes": "RAS"}),
                           ("spacing", {'pixdim': (1.0, 1.0, 1.0), "mode": ("bilinear", "nearest")}),
                           # ("crop_foreground", {"source_key": "img", "mode": "constant", "margin": 5}),
                           ("normalizeintensity", {"nonzero": True}), ]

        if len(self.input_images) == 1:  # add channel dimension if only one input channel
            base_list_start.insert(("add_channel_img", {}), 1)

        list_of_transforms = list()

        if mode == "predict":
            base_list_start += [("to_tensor", {})]
            for t_name, kwargs in base_list_start:
                if self.verbose: list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
                list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
                if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))
            return list_of_transforms

        if self.patch_size != self.image_size:  # patching if needed
            base_list_start += [("crop_by_pos_neg", {"pos": 1, "neg": 3, "label_key": "seg",
                                                     "spatial_size": self.patch_size, "num_samples": self.num_samples})]

        base_list_end = [("bounding_boxes_generator", {"segmentation_mode": self.segmentation_mode,
                                                       "thresholds": self.seg_thresholds,
                                                       "n_classes": self.n_classes}),
                         # ("filter_empty_images", {}),
                         ("to_tensor", {}),
                         ]

        # First add the images, reorient them and make sure the channel is first
        for t_name, kwargs in base_list_start:
            if self.verbose: list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
            list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
            if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))

        # Perform of augmentations if specified and if training mode
        if self.augmentations is not None and mode == "train":
            for transform in self.augmentations:
                if type(transform) != tuple:
                    transform = (transform, {})
                t_name, kwargs = transform

                if self.verbose:
                    list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
                    list_of_transforms.append(Lambdad(keys=["subject"], func=Printer(t_name)))
                list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
                if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))

        # Add the final transformations"#3k_64_n1-5_s6-14"
        for t_name, kwargs in base_list_end:
            if self.verbose: list_of_transforms.append(Lambdad(keys=["seg"], func=Printer(t_name)))
            list_of_transforms.append(get_transform_from_name(t_name, **kwargs))

        return list_of_transforms

    def setup(self, stage=None, perform_split=True, fold=None, n_splits=4):
        DS = CacheDataset if self.cache else Dataset
        self.fold = fold
        
        if self.subject is not None:
            self.trainsubs, self.valsubs = [self.subject], [self.subject]
            data_train = [
                {'img': [pjoin(self.data_dir, "imagesTr", f"{s}_{iid}.nii.gz") for iid in self.input_image_ids],
                 'seg': pjoin(self.data_dir, "labelsTr", f"{s}_{self.seg_filename}.nii.gz"),
                 'subject': s} for s in self.trainsubs]

            data_val = [
                {'img': [pjoin(self.data_dir, "imagesTr", f"{s}_{iid}.nii.gz") for iid in self.input_image_ids],
                 'seg': pjoin(self.data_dir, "labelsTr", f"{s}_{self.seg_filename}.nii.gz"),
                 'subject': s} for s in self.valsubs]
            
            self.train_dataset = DS(data=data_train, transform=self.train_transform)
            self.val_dataset = DS(data=data_val, transform=self.test_transform)
            return

        if self.percentage == -1:
            random_subj = randint(0, len(self.all_trainsubs))
            print("Picked subject", self.all_trainsubs[random_subj])
            self.trainsubs, self.valsubs = [self.all_trainsubs[random_subj]], [self.all_trainsubs[random_subj]]
            # print(self.trainsubs, self.testsubs)

        elif perform_split:
            if self.fold is not None:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                spl = list(kf.split(self.all_trainsubs))

                train_indices = [train_index for train_index, _ in spl]
                val_indices = [val_index for _, val_index in spl]

                self.trainsubs = self.all_trainsubs[train_indices[self.fold]]
                self.valsubs = self.all_trainsubs[val_indices[self.fold]]

            else:  # use default split (/!\ not desirable)
                warnings.warn("Non-desirable behavior: because no fold was specified, using test set as validation set")
                self.trainsubs, self.valsubs = self.all_trainsubs, self.testsubs

            data_train = [
                {'img': [pjoin(self.data_dir, "imagesTr", f"{s}_{iid}.nii.gz") for iid in self.input_image_ids],
                 'seg': pjoin(self.data_dir, "labelsTr", f"{s}_{self.seg_filename}.nii.gz"),
                 'subject': s} for s in self.trainsubs]

            data_test = [
                {'img': [pjoin(self.data_dir, "imagesTs", f"{s}_{iid}.nii.gz") for iid in self.input_image_ids],
                 'seg': pjoin(self.data_dir, "labelsTs", f"{s}_{self.seg_filename}.nii.gz"),
                 'subject': s} for s in self.testsubs]

            if self.fold is None:
                data_val = [
                    {'img': [pjoin(self.data_dir, "imagesTs", f"{s}_{iid}.nii.gz") for iid in self.input_image_ids],
                     'seg': pjoin(self.data_dir, "labelsTs", f"{s}_{self.seg_filename}.nii.gz"),
                     'subject': s} for s in self.valsubs]
            else:
                data_val = [
                    {'img': [pjoin(self.data_dir, "imagesTr", f"{s}_{iid}.nii.gz") for iid in self.input_image_ids],
                     'seg': pjoin(self.data_dir, "labelsTr", f"{s}_{self.seg_filename}.nii.gz"),
                     'subject': s} for s in self.valsubs]

            if stage == "fit" or stage is None:
                self.train_dataset = DS(data=data_train, transform=self.train_transform)
                self.val_dataset = DS(data=data_val, transform=self.test_transform)

            elif stage == "test" or stage is None:
                self.test_dataset = DS(data=data_test, transform=self.test_transform)

            elif stage == "predict" or stage is None:
                self.predict_train_dataset = DS(data=data_train, transform=self.predict_transform)
                self.predict_val_dataset = DS(data=data_val, transform=self.predict_transform)
                self.predict_test_dataset = DS(data=data_test, transform=self.predict_transform)

            else:
                raise ValueError(f"Stage '{stage}' not recognized")

        else:  # If we don't want to perform a split in predict mode, we use the whole dataset
            assert stage == "predict", "If you don't want to perform a split, you must be in predict mode"

            self.predictsubs = self.all_trainsubs + self.testsubs

            data_predict = [
                {'img': [pjoin(self.data_dir, self.predict_subdir, f"{s}_{iid}.nii.gz") for iid in
                         self.input_image_ids],
                 'subject': s} for s in self.valsubs]

            self.predict_dataset = DS(data=data_predict, transform=self.predict_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def predict_test_dataloader(self):
        return DataLoader(self.predict_test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_predict)

    def predict_val_dataloader(self):
        return DataLoader(self.predict_val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_predict)

    def predict_train_dataloader(self):
        return DataLoader(self.predict_train_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_predict)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_predict)


class ExampleDataset(pl.LightningDataModule):
    def __init__(self,
                 n_classes=1,
                 objects="multiple",
                 percentage=1.,
                 augmentations=None,
                 batch_size=8,
                 num_workers=int(os.cpu_count() / 2),
                 verbose=False,
                 show=False,
                 random_state=970205,
                 cache=False,
                 subject=None,
                 dataset_name=None,
                 seg_filename="seg",
                 segmentation_mode="classes",
                 seg_thresholds=None,
                 image_size=None,
                 patch_size=None,
                 num_samples=1,
                 data_dir="/home/wynen/PycharmProjects/MSLesions3D/data/artificial_dataset"):

        super().__init__()

        self.train_dataset = None
        self.test_dataset = None
        self.predict_train_dataset = None
        self.predict_test_dataset = None
        assert n_classes == 1 or n_classes == 2

        self.data_dir = data_dir
        self.data_dir = self.data_dir + r"/multiple_objects" if objects == "multiple" else self.data_dir

        self.data_dir = pjoin(self.data_dir, "one_class") if n_classes == 1 else \
            pjoin(self.data_dir, "double_class")
        self.data_dir = self.data_dir if dataset_name is None else pjoin(self.data_dir, dataset_name)

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.random_state = random_state
        self.cache = cache
        self.percentage = percentage
        self.verbose = verbose
        self.segmentation_mode = segmentation_mode
        self.seg_thresholds = seg_thresholds
        if self.segmentation_mode == "instances" and seg_thresholds is None:
            self.seg_thresholds = [(1, np.inf)]
        self.n_classes = n_classes
        self.augmentations = augmentations
        self.show = show
        self.subjects_list = [s.replace("sub-", "")[:4] for s in os.listdir(pjoin(self.data_dir, "images")) if
                              "sub-" in s]
        self.subjects_list = self.subjects_list[:int(percentage * len(self.subjects_list))] if percentage > 0 \
            else self.subjects_list
        self.subject = subject
        self.seg_filename = seg_filename

        assert image_size or patch_size, "Either image_size or patch_size must be specified"
        self.image_size = image_size
        self.patch_size = patch_size if patch_size is not None else image_size

        self.train_transform_list = self.get_list_of_transforms("train")
        self.test_transform_list = self.get_list_of_transforms("test")
        self.predict_transform_list = self.get_list_of_transforms("predict")

        self.train_transform = Compose(self.train_transform_list)
        self.test_transform = Compose(self.test_transform_list)
        self.predict_transform = Compose(self.predict_transform_list)

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers}

    def get_list_of_transforms(self, mode):
        """
        Return list of transforms for train, test or predict datasets
        """
        base_list_start = [("load_image", {}),
                           ("add_channel", {}),
                           ("normalizeintensity", {"nonzero": True}), ]

        list_of_transforms = list()

        if mode == "predict":
            base_list_start += [("to_tensor", {})]
            for t_name, kwargs in base_list_start:
                if self.verbose: list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
                list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
                if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))
            return list_of_transforms

        if self.patch_size != self.image_size:  # patching if needed
            base_list_start += [("crop_by_pos_neg", {"pos": 1, "neg": 7, "label_key": "seg",
                                                     "spatial_size": self.patch_size, "num_samples": self.num_samples})]

        base_list_end = [("bounding_boxes_generator", {"segmentation_mode": self.segmentation_mode,
                                                       "thresholds": self.seg_thresholds,
                                                       "n_classes": self.n_classes}),
                         ("to_tensor", {}),
                         ]

        # First add the images, reorient them and make sure the channel is first
        for t_name, kwargs in base_list_start:
            if self.verbose: list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
            list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
            if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))

        # Perform of augmentations if specified and if training mode
        if self.augmentations is not None and mode == "train":
            for transform in self.augmentations:
                if type(transform) != tuple:
                    transform = (transform, {})
                t_name, kwargs = transform

                if self.verbose:
                    list_of_transforms.append(Lambdad(keys=["img"], func=Printer(t_name)))
                    list_of_transforms.append(Lambdad(keys=["subject"], func=Printer(t_name)))
                list_of_transforms.append(get_transform_from_name(t_name, **kwargs))
                if self.show: list_of_transforms.append(ShowImage(keys=["img", "seg"], dim=2, grid=True, msg=t_name))

        # Add the final transformations"#3k_64_n1-5_s6-14"
        for t_name, kwargs in base_list_end:
            if self.verbose: list_of_transforms.append(Lambdad(keys=["seg"], func=Printer(t_name)))
            list_of_transforms.append(get_transform_from_name(t_name, **kwargs))

        return list_of_transforms

    def setup(self, stage=None, perform_split=True):
        DS = CacheDataset if self.cache else Dataset

        if self.subject is not None:
            self.trainsubs, self.testsubs = [self.subject], [self.subject]

        elif self.percentage == -1:
            random_subj = randint(0, len(self.subjects_list))
            print("Picked subject", self.subjects_list[random_subj])
            self.trainsubs, self.testsubs = [self.subjects_list[random_subj]], [self.subjects_list[random_subj]]
            # print(self.trainsubs, self.testsubs)

        elif perform_split:
            spl = train_test_split(self.subjects_list, train_size=0.8, test_size=0.2, random_state=self.random_state)
            self.trainsubs, self.testsubs = spl

            data_train = [{'img': pjoin(self.data_dir, "images", f"sub-{s}_image.nii.gz"),
                           'seg': pjoin(self.data_dir, "labels", f"sub-{s}_{self.seg_filename}.nii.gz"),
                           'subject': s} for s in self.trainsubs]
            data_test = [{'img': pjoin(self.data_dir, "images", f"sub-{s}_image.nii.gz"),
                          'seg': pjoin(self.data_dir, "labels", f"sub-{s}_{self.seg_filename}.nii.gz"),
                          'subject': s} for s in self.testsubs]

            if stage == "fit" or stage is None:
                self.train_dataset = DS(data=data_train, transform=self.train_transform)
                self.test_dataset = DS(data=data_test, transform=self.test_transform)

            elif stage == "test" or stage is None:
                self.test_dataset = DS(data=data_test, transform=self.test_transform)

            elif stage == "predict" or stage is None:
                self.predict_train_dataset = DS(data=data_train, transform=self.predict_transform)
                self.predict_test_dataset = DS(data=data_test, transform=self.predict_transform)

            else:
                raise ValueError(f"Stage '{stage}' not recognized")

        else:  # If we don't want to perform a split in predict mode, we use the whole dataset
            assert stage == "predict", "If you don't want to perform a split, you must be in predict mode"

            self.predictsubs = self.subjects_list

            data_predict = [{'img': pjoin(self.data_dir, "images", f"sub-{s}_image.nii.gz"),
                             'subject': s} for s in self.predictsubs]

            self.predict_dataset = DS(data=data_predict, transform=self.predict_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def predict_test_dataloader(self):
        return DataLoader(self.predict_test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_predict)

    def predict_train_dataloader(self):
        return DataLoader(self.predict_train_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_predict)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_predict)


if __name__ == '__main__':
    pass
    ds = ObjectDetectionDataset(data_dir="/home/wynen/data/rimnet",
                                input_images=('FLAIR', 'acq-phase_T2star'),
                                seg_filename="bboxes_prl",
                                segmentation_mode="instances",
                                patch_size=(96, 96, 96),
                                verbose=False,
                                num_workers=1,
                                batch_size=3,
                                num_samples=7)
    ds.setup("fit")
    dl = ds.train_dataloader()
    for b in dl:
        break
    print(b)
