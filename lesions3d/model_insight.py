# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:21:07 2022

@author: Maxence Wynen
"""

import torch
import pytorch_lightning as pl
from ssd3d import *
# import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join as pjoin
from os.path import exists as pexists
from datasets import ExampleDataset
import nibabel as nib
from utils import gcxgcygcz_to_cxcycz, cxcycz_to_gcxgcygcz


def idk_what_this_does():
    model_path = r"C:\Users\Cristina\Desktop\MSLesions3D\tensorboard\lesions\zebardi\no_augmentation\checkpoints\epoch=249-step=3250.ckpt"

    # model = LSSD3D.load_from_checkpoint(checkpoint_path=model_path)
    model = LSSD3D(n_classes=1 + 1, input_channels=1, lr=5e-4, width_mult=0.4, scheduler="none", batch_size=8,
                   comments="comments", compute_metric_every_n_epochs=5)
    model.init()

    params_per_layer = {}
    zeros_per_layer = {}

    for n, p in model.named_parameters():

        part = n.split('.')[0]
        if "bias" in n:
            continue

        try:
            n_layer = n.split('.')[2]

            if part not in params_per_layer:
                params_per_layer[part] = {}
            if n_layer not in params_per_layer[part]:
                params_per_layer[part][n_layer] = list()

            params_per_layer[part][n_layer].extend(p.cpu().view(-1).tolist())
        except IndexError:
            print(n)
            params_per_layer[part] = p.cpu().view(-1).tolist()

    for part, dic in params_per_layer.items():
        if type(dic) == dict:
            for layer, data in dic.items():
                plt.hist(data, bins=50, color='b')
                plt.title(
                    f'{part} layer {layer} // # zeros = {len([abs(x) for x in data if x < 1e-15])} out of {len(data)}')
                plt.show()


def save_prior_boxes(loader, det_locs, output_dir=r"./predictions", filename=""):
    print("Saving prior boxes ...")
    if not pexists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            affine = batch["img_meta_dict"][0]["affine"]
            subj = batch["subject"][0]
            print("Subject", subj, end="...  ")

            det_locs_subj = det_locs[i]

            img_shape = batch["img"].squeeze().numpy().shape
            pred_seg_subj = np.zeros(img_shape)

            for j, det_box in enumerate(det_locs_subj):
                det_box = torch.clip(det_box, 0, 1)
                det_box *= torch.Tensor(img_shape * 2)
                det_box = det_box.numpy().astype(int).tolist()
                x_min, y_min, z_min, x_max, y_max, z_max = det_box

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                z_min = max(z_min, 0)
                x_max = min(x_max + 1, img_shape[0] - 1)
                y_max = min(y_max + 1, img_shape[1] - 1)
                z_max = min(z_max + 1, img_shape[2] - 1)

                pred_seg_subj[x_min, y_min:y_max, z_min:z_max] = j + 1
                pred_seg_subj[x_max, y_min:y_max, z_min:z_max] = j + 1

                pred_seg_subj[x_min:x_max, y_min, z_min:z_max] = j + 1
                pred_seg_subj[x_min:x_max, y_max, z_min:z_max] = j + 1

                pred_seg_subj[x_min:x_max, y_min:y_max, z_min] = j + 1
                pred_seg_subj[x_min:x_max, y_min:y_max, z_max] = j + 1

                pred_seg_subj[x_min:x_max, y_max, z_max] = j + 1
                pred_seg_subj[x_max, y_min:y_max, z_max] = j + 1
                pred_seg_subj[x_max, y_max, z_min:z_max] = j + 1

                pred_seg_subj[x_max, y_max, z_max] = j + 1

            print(np.unique(pred_seg_subj))

            nib_img = nib.Nifti1Image(pred_seg_subj, affine.squeeze())
            filename="_"+filename
            nib.save(nib_img, pjoin(output_dir, f"sub-{subj}_prior-boxes{filename}.nii.gz"))


def show_prior_boxes(output_dir=r"./predictions",
                     aspect_ratios={3: [1.], 5: [1.], 7: [1.]}, scales={1: 0.05, 3: 0.1, 5: .15, 7: .2}):
    """
    Show candidate bounding boxes according to aspect ratios and scales
    Args:
        output_dir: path to where the image has to be saved
        aspect_ratios: layers and prior aspect ratios
        scales:
    """
    pl.seed_everything(970205)

    dataset = ExampleDataset(n_classes=1, percentage=-1, cache=False, num_workers=8, objects="multiple", batch_size=1)
    dataset.setup(stage="predict")
    loader = dataset.predict_test_dataloader()

    model = LSSD3D(n_classes=2, input_channels=1, lr=0.0005, width_mult=0.4, scheduler=None, batch_size=1,
                   input_size=(64, 64, 64), compute_metric_every_n_epochs=5, use_wandb=False,
                   ASPECT_RATIOS=aspect_ratios, SCALES=scales, min_score=0, top_k=50000)
    model.init()

    if not pexists(output_dir): os.makedirs(output_dir)

    predictor = pl.Trainer(accelerator="gpu", devices=1, enable_progress_bar=True)
    predictions_all_batches = predictor.predict(model, dataloaders=loader)

    model.to(device)

    det_locs = list()
    det_labels = list()
    det_scores = list()
    for locs, labels, scores in predictions_all_batches:
        det_locs.append(locs[0])  # first element because batch size is 1
        det_labels.append(labels[0])
        det_scores.append(scores[0])

    # save_prior_boxes(loader, det_locs, output_dir, filename=f"hellooo")

    prior_boxes_per_feature_map = model.create_prior_boxes(per_feature_map=True)

    for fmap, prior_boxes in prior_boxes_per_feature_map.items():
        print(f"fmap = {fmap}")
        prior_boxes = [torch.stack([torch.FloatTensor(prior_box) for prior_box in prior_boxes ])]
        decoded_locs = cxcycz_to_xyz(gcxgcygcz_to_cxcycz(torch.zeros((prior_boxes[0].shape[0],6)), prior_boxes[0]))
        save_prior_boxes(loader, [decoded_locs], output_dir, filename=f"fmap-{fmap}")


if __name__ == "__main__":
    sc1 = {1: 0.05, 3: 0.1, 5: .15, 7: .2}
    sc2 = {1: 0.3, 3: 0.5, 5: .7, 7: .9}
    show_prior_boxes(output_dir=r'/home/wynen/MSLesions3D/data/predictions', scales=sc1)
