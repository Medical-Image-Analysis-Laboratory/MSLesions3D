import torch
# import json
import os
# from os.path import exists as pexists
from os.path import join as pjoin
from utils import BoundingBoxesGeneratord, make_segmentation_from_bboxes, ShowImage
from random import randint
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from copy import deepcopy
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    # Spacingd,
    # ScaleIntensityRanged,
    CropForegroundd,
    # Resized,
    Orientationd,
    # EnsureChannelFirstd,
    # CopyItemsd,
    # SpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    SaveImaged,
    Lambdad,
    InvertibleTransform
)
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datasets import collate_fn




if __name__ == "__main__":

    slice = 0.666
    slice_after = 0.62
    dim = 2
    transform = Compose([
                    LoadImaged(keys=["img", "seg"]),
                    # ShowImage(keys=['seg'], slice=slice, dim=dim, msg="LoadImaged"),
                    AddChanneld(keys=["img","seg"]),
                    # ShowImage(keys=['seg'], slice=slice, dim=dim, msg="AddChanneld"),
                    ScaleIntensityd(keys=['img']),
                    # ShowImage(keys=['seg'], slice=slice, dim=dim, msg="ScaleIntensityd"),
                    ResizeWithPadOrCropd(keys=["img",'seg'], spatial_size=(400,400,400)),
                    # ShowImage(keys=['seg'], slice=slice_after, dim=dim, msg="ResizeWithPadOrCropd"),
                    BoundingBoxesGeneratord(keys=["seg"], segmentation_mode="classes", n_classes=1),
                    # ShowImage(keys=['seg'], slice=slice_after, dim=dim, msg="BoundingBoxesGeneratord"),
                    ToTensord(keys=["img"]),
                    # MakePredictionsd(keys=["pred_labels","pred_boxes"]),
                    ShowImage(keys=['img', 'seg'], slice=slice_after, dim=dim, msg="LastStep", grid=True)
                ])

    data_dir = r"C:\Users\Cristina\Desktop\MSLesions3D\data\example\multiple_objects\one_class"
    trainsubs = ["0000", "0001", "0002", "0003"]
    data = [{'img': pjoin(data_dir, "images", f"sub-{s}_image.nii.gz"),
           'seg': pjoin(data_dir, "labels", f"sub-{s}_seg.nii.gz"),
           # 'pred_labels': pjoin(data_dir, "labels", f"sub-{s}_seg.nii.gz"),
           # 'pred_boxes': pjoin(data_dir, "labels", f"sub-{s}_seg.nii.gz"),
           'subject': s} for s in trainsubs]

    dataset = Dataset(data=data, transform=transform)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


    for i, batch in enumerate(loader):
        pass
        # batch["boxes"] += 0.01
        pred_boxes, pred_labels = make_segmentation_from_bboxes(batch["boxes"], batch["labels"], (400,400,400))

        batch["seg"] = pred_boxes
        inversed_batch = [transform.inverse(x) for x in decollate_batch(batch)]
        preds_boxes = [SaveImaged(keys="seg", output_dir=data_dir, output_postfix="pred_boxes")(x) for x in inversed_batch]
        orig_image = [SaveImaged(keys="img", output_dir=data_dir, output_postfix="orig")(x) for x in inversed_batch]
        #
        #
        # batch["seg"] = pred_labels
        # inversed_batch = [transform.inverse(x) for x in decollate_batch(batch)]
        # preds_labels = [SaveImaged(keys="seg", output_dir=data_dir, output_postfix="pred_labels")(x) for x in inversed_batch]

