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


# warnings.filterwarnings("ignore", category=SyntaxWarning)
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


def compute_subjects_mAP(model, loader = None, dataset=None, subject_id = None, path_to_dir=r"./predictions"):
    # Expects Loader to have batch size = 1

    assert loader or (subject_id and dataset)

    def compute_subjects_metrics(subject, multiple=False):
        images, [gt_boxes, gt_labels] = subject["img"], subject['seg']

        images = images.to(device)

        # if multiple:
        #     gt_boxes = gt_boxes[0]
        #     gt_labels = gt_labels[0]

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
                APs, mAP, _ = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties)
                mAP = torch.FloatTensor([mAP])
            else:
                mAP = torch.FloatTensor([-10])
                print("Couldn't compute mAP for subject {subject['subject']}")

        return APs, float(mAP.cpu())

    if subject_id:
        for s in dataset.predict_test_dataset.data:
            if s["subject"] != subject_id:
                pass
            subject = s
            break
        return compute_subjects_metrics(subject)

    else:
        all_metrics = {}
        for subject in loader:
            all_metrics[subject["subject"][0]] = compute_subjects_metrics(subject, multiple=True)

        with open(pjoin(path_to_dir, f"aa_metrics.json"), "w") as json_file:
            json.dump(all_metrics, json_file)

        return all_metrics




def save_predictions_example(loader, det_locs, det_labels, det_scores, path_to_dir=r"./predictions"):
    print("Saving predictions ...")

    if not pexists(path_to_dir):
        os.makedirs(path_to_dir)

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
            if len(det_locs_subj) == 1 and \
                    (det_locs_subj == torch.Tensor([[0., 0., 0., 1., 1., 1.]])).sum() == 6:
                print("# predicted boxes: 0 ;", "# of GT boxes:", len(batch['boxes'][0]))
            else:
                print("# predicted boxes:", len(det_locs_subj), ";", "# of GT boxes:", len(batch['boxes'][0]))

            for j, det_box in enumerate(det_locs_subj):
                det_box_frac = list(det_box.detach().numpy().astype(float))
                det_label = int(det_labels_subj[j])
                if det_label == 0:
                    continue
                det_box = torch.clip(det_box, 0, 1)
                det_box *= torch.Tensor(img_shape * 2)
                det_box = det_box.numpy().astype(int)
                det_box = det_box.tolist()  # (det_box - np.array([0,0,0,1,1,1])).tolist()
                x_min, y_min, z_min, x_max, y_max, z_max = det_box

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                z_min = max(z_min, 0)
                x_max = min(x_max, img_shape[0] - 1)
                y_max = min(y_max, img_shape[1] - 1)
                z_max = min(z_max, img_shape[2] - 1)

                pred_seg_subj[x_min, y_min:y_max, z_min:z_max] = j + 1
                pred_seg_subj[x_max, y_min:y_max, z_min:z_max] = j + 1

                pred_seg_subj[x_min:x_max, y_min, z_min:z_max] = j + 1
                pred_seg_subj[x_min:x_max, y_max, z_min:z_max] = j + 1

                pred_seg_subj[x_min:x_max, y_min:y_max, z_min] = j + 1
                pred_seg_subj[x_min:x_max, y_min:y_max, z_max] = j + 1

                det_score = det_scores_subj[j]
                scores_map.append((j + 1, det_score))

                all_infos[j + 1] = (det_box_frac, det_box, det_label, float(det_score))

            nib_img = nib.Nifti1Image(pred_seg_subj, affine.squeeze())
            nib.save(nib_img, pjoin(path_to_dir, f"sub-{subj}_preds.nii.gz"))

            infos = pd.DataFrame(scores_map, columns=["label_id", "score"])
            infos.to_csv(pjoin(path_to_dir, f"sub-{subj}_preds.csv"))

            with open(pjoin(path_to_dir, f"sub-{subj}_preds.json"), "w") as json_file:
                json.dump(all_infos, json_file)


def predict_example():
    pl.seed_everything(970205)

    n_classes = 1

    path = r"C:\Users\Cristina\Desktop\MSLesions3D\tensorboard\example\full_dataset_400_100\version_9\checkpoints\epoch=145-step=7300.ckpt"

    model = LSSD3D.load_from_checkpoint(path, min_score=0.5).to(device)

    dataset = ExampleDataset(n_classes=n_classes, percentage=0.025, cache=False, num_workers=8, objects="multiple",
                             batch_size=1)

    path_to_dir = r"C:\Users\Cristina\Desktop\MSLesions3D\data\example\\multiple_objects\one_class\predictions"
    dataset.setup(stage="predict")
    loader = dataset.predict_train_dataloader()

    for data in dataset.predict_train_dataset.data: print(data["subject"])

    predictor = pl.Trainer(gpus=1, enable_progress_bar=True)
    predictions_all_batches = predictor.predict(model, dataloaders=loader)

    model = model.to(device)

    det_locs = list()
    det_labels = list()
    det_scores = list()
    for locs, labels, scores in predictions_all_batches:
        det_locs.append(locs[0])
        det_labels.append(labels[0])
        det_scores.append(scores[0])

    if path_to_dir is not None:
        save_predictions_example(loader, det_locs, det_labels, det_scores, path_to_dir)

    print(compute_subjects_mAP(model, loader=loader, dataset=None, subject_id=None, path_to_dir=path_to_dir))

    pass


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
    pl.seed_everything(970205)

    path = r""

    model = LSSD3D.load_from_checkpoint(path).to(device)

    dataset = LesionsDataModule(percentage=0.1,batch_size=1)
    dataset.setup(stage="predict")

    loader = dataset.predict_train_dataloader()
    for data in dataset.predict_train_dataset.data: print(data["subject"])

    predictor = pl.Trainer(gpus=1, enable_progress_bar=True)
    predictions_all_batches = predictor.predict(model, dataloaders=loader)

    model = model.to(device)

    det_locs = list()
    det_labels = list()
    det_scores = list()
    for locs, labels, scores in predictions_all_batches:
        det_locs.append(locs[0])
        det_labels.append(labels[0])
        det_scores.append(scores[0])

    path_to_dir = r"C:\Users\Cristina\Desktop\MSLesions3D\data\lesions\predictions"
    print(compute_subjects_mAP(model, loader=loader, dataset=None, subject_id=None, path_to_dir=path_to_dir))


