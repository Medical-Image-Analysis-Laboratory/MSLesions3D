# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:12:13 2022

@author: Maxence Wynen
Inspired by https://github.com/MIC-DKFZ/nnDetection/blob/main/scripts/generate_example.py
"""

import os
from os.path import join as pjoin
from os.path import exists as pexists
import random
from multiprocessing import Pool
from itertools import repeat
import nibabel as nib
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dim', type=int, required=False, default=3, help="number of dimensions for the images")
parser.add_argument('--n_classes', type=int, required=False, default=1, help="number of different classes to generate")
parser.add_argument('--image_size', type=int, nargs='+', default=[250, 300, 300],
                    help="image size (length must match the number of dimensions)")
parser.add_argument('--object_size', type=int, nargs='+', default=[10, 32], help="range for object size [min, max]")
parser.add_argument('--num_objects', type=int, nargs='+', default=[2, 5],
                    help="range of number of objects to add on the image")
parser.add_argument('--object_width', type=int, required=False, default=4, help="width if # classes is 2")
parser.add_argument('--num_processes', type=int, required=False, default=8, help="number of processes")
parser.add_argument('--num_images', type=int, required=False, default=500, help="number of images to generate")
parser.add_argument('--noise', type=int, default=1, help="whether to add noise to the image or not")
parser.add_argument('--output_dir', type=str, required=True,
                    default=rf"/home/wynen/MSLesions3D/data/artificial_dataset/", help="output directory")
parser.add_argument('--random_seed', type=int, default=0, help="random seed")
parser.add_argument('--box_noise', type=int, default=0, help="whether to add random noise the boxes intensity or not")

args = parser.parse_args()
dim = args.dim  # 3
image_size = list(args.image_size)  # [250, 300, 300]
object_size = sorted(list(args.object_size))  # [10, 32]
object_width = args.object_width  # 4
n_classes = args.n_classes  # 1
num_processes = args.num_processes  # 8
num_images = args.num_images  # 500
add_noise = bool(args.noise)
box_noise = bool(args.box_noise)
num_objects = args.num_objects
random_seed = args.random_seed
print(f"Random seed set at {random_seed}")

DIR = "one_class" if n_classes == 1 else "double_class"
DIR = "multiple_objects/one_class"

image_dir = pjoin(args.output_dir, DIR, "images")  # rf"/home/wynen/MSLesions3D/data/artificial_dataset/{DIR}/images"
seg_dir = pjoin(args.output_dir, DIR, "labels")  # rf"/home/wynen/MSLesions3D/data/artificial_dataset/{DIR}/segs"

if not pexists(image_dir):
    os.makedirs(image_dir)
if not pexists(seg_dir):
    os.makedirs(seg_dir)


def average_overlapping_intensity(image, mask, slicing, intensity, noise):
    """
    Average the intensity of the overlapping voxels in the image. If the mask is 1 at a given voxel in the slicing,
    the intensity of the image is averaged with the intensity of the box. If the mask is 0, the intensity of the image
    is set to intensity.
    :param image: image to modify
    :param mask: mask telling which voxels have been modified
    :param slicing: slicing of the new box
    :param intensity: intensity of the new box
    :return: modified image
    """
    image_slice = image[slicing]
    mask_slice = mask[slicing]
    overlap = mask_slice > 0 # Find overlapping voxels
    image_slice[overlap] = (image_slice[overlap] + intensity) / 2  # Average intensity of overlapping voxels
    image_slice[~overlap] = image_slice[~overlap] + 0.4 if noise else intensity   # Set intensity of non-overlapping voxels in mask to intensity
    image_slice[~overlap] = np.clip(image_slice[~overlap], 0, 1)  # Clip intensity to [0, 1]
    image[slicing] = image_slice
    return image


def generate_image(image_dir, seg_dir, idx, n_classes, noise=add_noise):
    print(f"Generating image and segmentation for case {idx}...")
    random.seed(random_seed + idx)
    np.random.seed(random_seed + idx)

    data = np.random.rand(*image_size) if noise else np.zeros(image_size)
    mask = np.zeros_like(data)
    borders = np.zeros_like(data)

    n_objects = np.random.randint(*num_objects)

    for obj_idx in range(n_objects + 1):

        selected_size = np.random.randint(object_size[0], object_size[1])
        selected_class = np.random.randint(0, n_classes)
        # Class 0: fully filled
        # Class 1: borders fully filled

        top_left = [np.random.randint(0, image_size[i] - selected_size) for i in range(dim)]

        if selected_class == 0:
            slicing = tuple([slice(tp, tp + selected_size) for tp in top_left])
            intensity = 1 if not box_noise else np.random.uniform(0.5, 1)
            data = average_overlapping_intensity(data, mask, slicing, intensity, noise)
            mask[slicing] = 1

            if dim == 2:
                borders[slicing[0].start:slicing[0].stop + 1, slicing[1].start] = obj_idx + 1
                borders[slicing[0].start:slicing[0].stop + 1, slicing[1].stop] = obj_idx + 1
                borders[slicing[0].start, slicing[0].start:slicing[1].stop] = obj_idx + 1
                borders[slicing[0].stop, slicing[0].start:slicing[1].stop] = obj_idx + 1
            elif dim == 3:
                borders[slicing[0].start:slicing[0].stop, slicing[1].start:slicing[1].stop, slicing[2].start] = obj_idx + 1
                borders[slicing[0].start:slicing[0].stop, slicing[1].start:slicing[1].stop, slicing[2].stop -1] = obj_idx + 1

                borders[slicing[0].start, slicing[1].start:slicing[1].stop, slicing[2].start:slicing[2].stop] = obj_idx + 1
                borders[slicing[0].stop -1, slicing[1].start:slicing[1].stop, slicing[2].start:slicing[2].stop] = obj_idx + 1

                borders[slicing[0].start:slicing[0].stop, slicing[1].start, slicing[2].start:slicing[2].stop] = obj_idx + 1
                borders[slicing[0].start:slicing[0].stop, slicing[1].stop -1, slicing[2].start:slicing[2].stop] = obj_idx + 1

        # Not handled
        elif selected_class == 1:
            slicing = tuple([slice(tp, tp + selected_size) for tp in top_left])

            inner_slicing = [slice(tp + object_width, tp + selected_size - object_width) for tp in top_left]
            if len(inner_slicing) == 3:
                inner_slicing[0] = slice(0, image_size[0])
            inner_slicing = tuple(inner_slicing)

            object_mask = np.zeros_like(mask).astype(bool)
            object_mask[slicing] = 1
            object_mask[inner_slicing] = 0

            intensity = 1 if not box_noise else np.random.uniform(0.5, 1)
            data[object_mask] = data[object_mask] + 0.4 if noise else intensity
            data = data.clip(0, 1)
            mask[object_mask] = 2
        else:
            raise NotImplementedError

    image = nib.Nifti1Image(data, affine=np.eye(4))
    seg = nib.Nifti1Image(mask, affine=np.eye(4))
    labels = nib.Nifti1Image(borders, affine=np.eye(4))

    nib.save(image, pjoin(image_dir, f"sub-{str(idx).zfill(4)}_image.nii.gz"))
    nib.save(seg, pjoin(seg_dir, f"sub-{str(idx).zfill(4)}_seg.nii.gz"))
    nib.save(labels, pjoin(seg_dir, f"sub-{str(idx).zfill(4)}_labels.nii.gz"))


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
    # generate_image(image_dir, seg_dir, num_images, n_classes)
    main()
