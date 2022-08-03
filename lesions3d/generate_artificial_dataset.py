# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:12:13 2022

@author: Maxence Wynen
Inspired by https://github.com/MIC-DKFZ/nnDetection/blob/main/scripts/generate_example.py
"""

import os
from os.path import join as pjoin
import random
from multiprocessing import Pool
from itertools import repeat
import nibabel as nib

import numpy as np


dim = 3
image_size = [250, 300, 300]
object_size = [16, 32]
object_width = 4
n_classes = 1
num_processes = 8
num_images = 500

DIR = "one_class" if n_classes == 1 else "double_class"
DIR = "multiple_objects\one_class"

image_dir = rf"C:\Users\Cristina\Desktop\MSLesions3D\data\example\{DIR}\images"
seg_dir = rf"C:\Users\Cristina\Desktop\MSLesions3D\data\example\{DIR}\labels"

def generate_image(image_dir, label_dir, idx, n_classes, noise=True):
    print(f"Generating image and segmentation for case {idx}...")
    random.seed(idx)
    np.random.seed(idx)
    
    data = np.random.rand(*image_size) if noise else np.zeros_like(data)
    mask = np.zeros_like(data)

    n_objects = np.random.randint(1,5)

    for _ in range(n_objects+1):

        selected_size = np.random.randint(object_size[0], object_size[1])
        selected_class = np.random.randint(0, n_classes)
        # Class 0: fully filled
        # Class 1: borders fully filled

        top_left = [np.random.randint(0, image_size[i] - selected_size) for i in range(dim)]

        if selected_class == 0:
            slicing = tuple([slice(tp, tp + selected_size) for tp in top_left])
            data[slicing] = data[slicing] + 0.4
            data = data.clip(0, 1)
            mask[slicing] = 1
        elif selected_class == 1:
            slicing = tuple([slice(tp, tp + selected_size) for tp in top_left])

            inner_slicing = [slice(tp + object_width, tp + selected_size - object_width) for tp in top_left]
            if len(inner_slicing) == 3:
                inner_slicing[0] = slice(0, image_size[0])
            inner_slicing = tuple(inner_slicing)

            object_mask = np.zeros_like(mask).astype(bool)
            object_mask[slicing] = 1
            object_mask[inner_slicing] = 0

            data[object_mask] = data[object_mask] + 0.4
            data = data.clip(0, 1)
            mask[object_mask] = 2
        else:
            raise NotImplementedError
        
    
    image = nib.Nifti1Image(data, affine=np.eye(4))
    seg = nib.Nifti1Image(mask, affine=np.eye(4))
    
    nib.save(image, pjoin(image_dir, f"sub-{str(idx).zfill(4)}_image.nii.gz"))
    nib.save(seg, pjoin(seg_dir, f"sub-{str(idx).zfill(4)}_seg.nii.gz"))

def main():
    with Pool(processes=num_processes) as p:
        p.starmap(
            generate_image,
            zip(
                repeat(image_dir),
                repeat(seg_dir),
                range(num_images),
                repeat(n_classes)
            )
        )

if __name__ == "__main__":
    main()

