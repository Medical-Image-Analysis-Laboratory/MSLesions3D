# from datasets import *
# from monai.data import box_area
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from os.path import join as pjoin
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = r"/usr/lib/x86_64-linux-gnu/qt5/plugins/"
# os.environ["QT_DEBUG_PLUGINS"] = "1"

data_dirs = ["/home/wynen/data/rimnet/labelsTr",
             "/home/wynen/data/rimnet/labelsTs"]


def get_bboxes(seg):
    bboxes = []
    for instance in np.unique(seg):
        if instance == 0:
            continue
        x, y, z = np.where(seg == instance)
        x1, x2, y1, y2, z1, z2 = min(x), max(x), min(y), max(y), min(z), max(z)
        bboxes.append(np.array([x1, x2, y1, y2, z1, z2]))
    if len(bboxes) == 0:
        return None
    return np.vstack(np.array(bboxes, dtype='object'))


def make_bboxes_dataset(data_dirs, prl_only=False):
    bboxes = []
    end = "_bboxes.nii.gz" if not prl_only else "_bboxes_prl.nii.gz"
    for data_dir in data_dirs:
        for seg_file in os.listdir(data_dir):
            if seg_file.endswith(end):
                print(seg_file)
                seg = nib.load(pjoin(data_dir, seg_file)).get_fdata()
                seg_bboxes = get_bboxes(seg)
                if seg_bboxes is not None:
                    bboxes.append(seg_bboxes)
    return np.vstack(np.array(bboxes, dtype='object'))



if __name__ == "__main__":
    pass
    # bboxes = make_bboxes_dataset(data_dirs)
    # np.save(r"/home/wynen/bboxes.npy", bboxes)
    # bboxes = make_bboxes_dataset(data_dirs, prl_only=True)
    # np.save(r"/home/wynen/bboxes_prl.npy", bboxes)

    # with open("/home/wynen/bboxes.npy", "rb") as f:
    #     bboxes = np.load(f, allow_pickle=True)
    with open("/home/wynen/bboxes_prl.npy", "rb") as f:
        bboxes = np.load(f, allow_pickle=True)

    lengths = list()
    widths = list()
    depths = list()
    volumes = list()
    for bbox in bboxes:
        length = (bbox[1] - bbox[0])
        width = (bbox[3] - bbox[2])
        depth = (bbox[5] - bbox[4])
        volume = (length*width*depth)

        if volume != 0:

            lengths += [length]
            widths += [depth]
            depths += [depth]
            volumes += [volume]

    sides = lengths + widths + depths

    print(f"Mean length: {np.mean(lengths)}")
    print(f"Median length: {np.median(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Std length: {np.std(lengths)}")

    print(f"Mean width: {np.mean(widths)}")
    print(f"Median width: {np.median(widths)}")
    print(f"Max width: {np.max(widths)}")
    print(f"Min width: {np.min(widths)}")
    print(f"Std width: {np.std(widths)}")

    print(f"Mean depth: {np.mean(depths)}")
    print(f"Median depth: {np.median(depths)}")
    print(f"Max depth: {np.max(depths)}")
    print(f"Min depth: {np.min(depths)}")
    print(f"Std depth: {np.std(depths)}")

    print(f"Mean volume: {np.mean(volumes)}")
    print(f"Median volume: {np.median(volumes)}")
    print(f"Max volume: {np.max(volumes)}")
    print(f"Min volume: {np.min(volumes)}")
    print(f"Std volume: {np.std(volumes)}")

    print(f"Mean side: {np.mean(sides)}")
    print(f"Median side: {np.median(sides)}")
    print(f"Max side: {np.max(sides)}")
    print(f"Min side: {np.min(sides)}")
    print(f"Std side: {np.std(sides)}")

    volume_threshold = 1.5e4
    volumes = np.array(volumes)
    print(f"Number of boxes with volume > {volume_threshold}: {np.sum(volumes > volume_threshold)}")
    print(f"Number of boxes with volume <= {volume_threshold}: {np.sum(volumes <= volume_threshold)}")
    volumes = volumes[volumes <= volume_threshold]

    # plt.hist(sides, bins=50)
    # plt.title("Sides")
    # plt.show()
    # plt.hist(lengths, bins=50)
    # plt.title("Lengths")
    # # plt.xlim((0,0.2))
    # plt.show()
    # plt.hist(widths, bins=50)
    # plt.title("Widths")
    # # plt.xlim((0,0.2))
    # plt.show()
    # plt.hist(depths, bins=50)
    # plt.title("Depths")
    # # plt.xlim((0,0.2))
    # plt.show()
    # plt.hist(volumes, bins=50)
    # plt.title(f"Volumes")
    # # plt.xlim((0,volume_threshold))
    # plt.show()
    plt.hist(volumes, bins=50)
    plt.title(f"Volumes < {volume_threshold}")
    plt.xlim((0,volume_threshold))
    plt.show()
    pass







