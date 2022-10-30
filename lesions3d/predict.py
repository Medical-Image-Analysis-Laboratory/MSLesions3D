# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:30:58 2022

@author: Maxence Wynen
"""

from datasets import *
from ssd3d import LSSD3D
import pytorch_lightning as pl
import nibabel as nib
import numpy as np
import torch
import os
from os.path import join as pjoin
from os.path import exists as pexists
import pandas as pd
from utils import calculate_mAP
import warnings
import json
from utils import make_segmentation_from_bboxes
from monai.data import decollate_batch
from monai.transforms import SaveImaged
import argparse
from pprint import pprint
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset_path', type=str, help="path to dataset used for training and validation",
                    default=r'../data/artificial_dataset')
parser.add_argument('-dn', '--dataset_name', type=str, help="name of dataset to use", default=None)
parser.add_argument('-m', '--model_path', type=str, help="path to model", default=r'model_final.onnx')
parser.add_argument('-mn', '--model_name', type=str, help="wandb model name", default=None)
parser.add_argument('-p', '--percentage', type=float, help="percentage of the dataset to predict on", default=1.)
parser.add_argument('-su', '--subject', type=str, default=None, help="if prediction has to be done on 1 subject only, specify its id")
parser.add_argument('-c', '--n_classes', type=int, help="number of classes in dataset", default=1)
parser.add_argument('-nw', '--num_workers', type=int, default=8, help="number of workers for the dataset")
parser.add_argument('-ps', '--predict_subset', type=str, help="subset to predict on", choices=['train', 'validation', 'test', 'all'], default=r'train')
parser.add_argument('-sc', '--min_score', type=float, help="minimum score for a candidate box to be considered as positive in the visualisation", default=0.5) #0.5
parser.add_argument('-k', '--top_k', type=int, help="if there are a lot of resulting detection across all classes, keep only the top 'k'", default=100) #100
parser.add_argument('-o', '--output_dir', type=str, help="path to output", default=r"../data/predictions/")
parser.add_argument('-si', '--save_images', type=int, help="whether to save the predictions as nii.gz images or not", default=1)
args = parser.parse_args()



def predict_all(dataset, model, predict_subset="test", min_score=0.5, max_overlap=0.5, top_k=100, path_to_dir=None):
    """
    

    Args:
        dataset (TYPE): DESCRIPTION.
        model (TYPE): DESCRIPTION.
        predict_subset (TYPE, optional): train or test. Defaults to "train".
        min_score (TYPE, optional): DESCRIPTION. Defaults to 0.5.
        max_overlap (TYPE, optional): DESCRIPTION. Defaults to 0.5.
        top_k (TYPE, optional): DESCRIPTION. Defaults to 100.

    Returns:
        None.

    """
    dataset.batch_size = 1
    dataset.setup(stage="predict")
    loader = dataset.predict_train_dataloader() if predict_subset == "train" \
        else dataset.predict_test_dataloader()

    predictor = pl.Trainer(gpus=1, enable_progress_bar=True)
    predictions_all_batches = predictor.predict(model, dataloaders=loader)

    det_locs = list()
    det_labels = list()
    det_scores = list()

    for locs, labels, scores in predictions_all_batches:
        det_locs.append(locs[0])
        det_labels.append(labels[0])
        det_scores.append(scores[0])

    if path_to_dir is not None:
        save_predictions(loader, det_locs, det_labels, det_scores, path_to_dir)

    return det_locs, det_labels, det_scores


def compute_subjects_mAP(model, loader = None, dataset=None, subject_id = None, output_dir=r"./predictions", min_iou=0.5):
    # Expects Loader to have batch size = 1

    assert loader or (subject_id and dataset)

    def compute_subjects_metrics(subject, multiple=False):
        def convert_tensor(tensor):
            try:
                return tensor.cpu().detach().item()
            except ValueError:
                return tensor.cpu().detach().tolist()

        images, [gt_boxes, gt_labels] = subject["img"], subject['seg']

        images = images.to(device)

        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)
        if multiple:
            images = images[0]

        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, model.min_score,
                                                                 model.max_overlap, model.top_k)
        with torch.no_grad():
            gt_difficulties = [torch.BoolTensor([False] * lbls.size(0)).to(device) for lbls in gt_labels]
            compute_mAP = [1 if len(predicted_locs_img) > 500 else 0 for predicted_locs_img in predicted_locs]
            compute_mAP = sum(compute_mAP)
            if compute_mAP != 0:
                # APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties, min_overlap=min_iou)
                # mAP = torch.FloatTensor([mAP])
                out = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties, min_overlap=min_iou, return_detail=True)
                # pprint(out)
                metrics = {}
                for key, value in out.items():
                    if type(value) in (float, int):
                        metrics[key] = value
                    elif type(value) == dict:
                        metrics[key] = {k: convert_tensor(v) for k,v in value.items()}
                    else:
                        metrics[key] = convert_tensor(value)
            else:
                mAP = torch.FloatTensor([-10])
                print("Couldn't compute mAP for subject {subject['subject']}")

        # return APs, float(mAP.cpu())
        # pprint(metrics)
        return metrics

    if subject_id:
        for s in dataset.predict_test_dataset.data:
            if s["subject"] != subject_id: pass
            subject = s
            break
        return compute_subjects_metrics(subject)

    else:
        all_metrics = {}
        for subject in loader: all_metrics[subject["subject"][0]] = compute_subjects_metrics(subject, multiple=True)

        with open(pjoin(output_dir, f"aa_metrics_per_subject_(min_IoU={min_iou}).json"), "w") as json_file:
            json.dump(all_metrics, json_file, indent=4)

        return all_metrics


def save_predictions_example(loader, det_locs, det_labels, det_scores, min_score=0.5,
                             output_dir=r"./predictions", save_images=True):
    print("Saving predictions ...")

    if not pexists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # original_affine = batch["img_meta_dict"][0]["original_affine"]
            affine = batch["img_meta_dict"][0]["affine"]
            subj = batch["subject"][0]
            print("Subject", subj, end="...  ")

            det_locs_subj = det_locs[i]
            det_labels_subj = det_labels[i]
            det_scores_subj = det_scores[i]

            assert det_locs_subj.shape[0] == det_labels_subj.shape[0] == det_labels_subj.shape[0]

            img_shape = batch["img"].squeeze().numpy().shape
            pred_seg_subj = np.zeros(img_shape)

            scores_map = []
            all_infos = {}
            print("img_shape:", img_shape, end=" ; ")
            if len(det_locs_subj) == 1 and (det_locs_subj == torch.Tensor([[0., 0., 0., 1., 1., 1.]])).sum() == 6:
                print("# predicted boxes: 0 ;", "# of GT boxes:", len(batch['boxes'][0]))
            else:
                print("# predicted boxes:", len(det_locs_subj), ";", "# of GT boxes:", len(batch['boxes'][0]))

            for j, det_box in enumerate(det_locs_subj):
                det_score = det_scores_subj[j]
                scores_map.append((j + 1, det_score))
                if det_score < min_score:
                    continue
                det_box_frac = list(det_box.detach().numpy().astype(float))
                det_label = int(det_labels_subj[j])
                if det_label == 0:
                    continue
                det_box = torch.clip(det_box, 0, 1)
                det_box *= torch.Tensor(img_shape * 2)
                det_box = det_box.numpy().astype(int).tolist()
                x_min, y_min, z_min, x_max, y_max, z_max = det_box

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                z_min = max(z_min, 0)
                x_max = min(x_max+1, img_shape[0] - 1)
                y_max = min(y_max+1, img_shape[1] - 1)
                z_max = min(z_max+1, img_shape[2] - 1)

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

                all_infos[j + 1] = (det_box_frac, det_box, det_label, float(det_score))

            if save_images:
                nib_img = nib.Nifti1Image(pred_seg_subj, affine.squeeze())
                nib.save(nib_img, pjoin(output_dir, f"sub-{subj}_preds.nii.gz"))

            infos = pd.DataFrame(scores_map, columns=["label_id", "score"])
            infos.to_csv(pjoin(output_dir, f"sub-{subj}_preds.csv"))

            with open(pjoin(output_dir, f"sub-{subj}_preds.json"), "w") as json_file:
                json.dump(all_infos, json_file)


def predict_example(model_path, output_dir, dataset_path, dataset_name, n_classes=1, subject=None, percentage=1.,
                    predict_subset="train", min_score=0.5, top_k=10, num_workers=8, save_images=True, model_name=None):
    pl.seed_everything(970205)

    path = model_path
    output_dir = output_dir if dataset_name is None else pjoin(output_dir, dataset_name)
    output_dir = output_dir if model_name is None else pjoin(output_dir, model_name)
    if not pexists(output_dir): os.makedirs(output_dir)
    shutil.copy(model_path, pjoin(output_dir, Path(model_path).name))
    output_dir = pjoin(output_dir, f"{predict_subset}_set")
    output_dir = pjoin(output_dir, f"min_score_{min_score}")
    if not pexists(output_dir): os.makedirs(output_dir)

    dataset = ExampleDataset(n_classes=n_classes, subject=subject, percentage=percentage,
                             cache=False, num_workers=num_workers, objects="multiple",
                             batch_size=1, data_dir=dataset_path, dataset_name=dataset_name)
    dataset.setup(stage="predict")
    if predict_subset == "train":
        loader = dataset.predict_train_dataloader()
    else:
        loader = dataset.predict_test_dataloader()

    model = LSSD3D.load_from_checkpoint(path, min_score=min_score).to(device)

    model.top_k = top_k
    model.min_score = min_score

    predictor = pl.Trainer(accelerator="gpu", devices=1, enable_progress_bar=True)
    predictions_all_batches = predictor.predict(model, dataloaders=loader)

    model.to(device)

    det_locs = list()
    det_labels = list()
    det_scores = list()
    for locs, labels, scores in predictions_all_batches:
        det_locs.append(locs[0]) # first element because batch size is 1
        det_labels.append(labels[0])
        det_scores.append(scores[0])

    if save_images and output_dir is not None:
        save_predictions_example(loader, det_locs, det_labels, det_scores, min_score, output_dir, save_images)

    print(f"\n\n_________________________AP for IoU = 0.5 / min score = {min_score}_________________________\n")
    pprint(compute_subjects_mAP(model, loader=loader, dataset=None, subject_id=None, output_dir=output_dir, min_iou=0.5))
    print(f"\n\n_________________________AP for IoU = 0.1 / min score = {min_score}_________________________\n")
    pprint(compute_subjects_mAP(model, loader=loader, dataset=None, subject_id=None, output_dir=output_dir, min_iou=0.1))


def save_predictions(dataset, loader, det_locs, det_labels, det_scores, path_to_dir):
    print("Saving predictions ...")

    if not pexists(path_to_dir):
        os.makedirs(path_to_dir)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            det_locs_subj = det_locs[i]
            det_labels_subj = det_labels[i]
            det_scores_subj = det_scores[i]

            assert det_locs_subj.shape[0] == det_labels_subj.shape[0] == det_labels_subj.shape[0]

            pred_boxes, pred_labels = make_segmentation_from_bboxes(det_locs_subj, det_labels_subj, (250,300,300))

            batch["seg"] = pred_boxes
            inversed_batch = [dataset.transform.inverse(x) for x in decollate_batch(batch)]
            preds_boxes = [SaveImaged(keys="seg", output_dir=data_dir, output_postfix="pred_boxes")(x) for x in
                           inversed_batch]
            # orig_image = [SaveImaged(keys="img", output_dir=data_dir, output_postfix="orig")(x) for x in inversed_batch]




if __name__ == "__main__":
    pass
    if args.predict_subset == 'all':
        for psubset in ["train", "validation", "test"]:
            predict_example(model_path=args.model_path, output_dir=args.output_dir, dataset_path=args.dataset_path,
                            dataset_name=args.dataset_name, n_classes=args.n_classes, subject=args.subject,
                            percentage=args.percentage, predict_subset=psubset,
                            min_score=args.min_score, top_k=args.top_k, num_workers=args.num_workers,
                            save_images=args.save_images, model_name=args.model_name)
    else:
        predict_example(model_path=args.model_path, output_dir=args.output_dir, dataset_path=args.dataset_path,
                        dataset_name=args.dataset_name, n_classes=args.n_classes, subject=args.subject,
                        percentage=args.percentage, predict_subset=args.predict_subset,
                        min_score=args.min_score, top_k=args.top_k, num_workers=args.num_workers,
                        save_images=args.save_images, model_name=args.model_name)


