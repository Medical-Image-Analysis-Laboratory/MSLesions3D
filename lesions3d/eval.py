# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:12:45 2022

@author: Maxence Wynen
"""

import json
from ssd3d import *
from datasets import *
from os.path import join as pjoin
import torch
from utils import calculate_mAP
from tqdm import tqdm
import argparse
#
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset_path', type=str, help="path to dataset used for training and validation",
                    default=r'../data/artificial_dataset')
parser.add_argument('-dn', '--dataset_name', type=str, help="name of dataset to use", default=None)
# parser.add_argument('-m', '--model_path', type=str, help="path to model",
#                     default=r'model_final.onnx')
parser.add_argument('-p', '--percentage', type=float, help="percentage of the dataset to predict on", default=1.)
# parser.add_argument('-su', '--subject', type=str, default=None,
#                     help="if prediction has to be done on 1 subject only, specify its id")
# parser.add_argument('-c', '--n_classes', type=int, help="number of classes in dataset", default=1)
# parser.add_argument('-nw', '--num_workers', type=int, default=8, help="number of workers for the dataset")
# parser.add_argument('-ps', '--predict_subset', type=str, help="subset to predict on", choices=['train', 'validation', 'test'], default=r'train')
# parser.add_argument('-sc', '--min_score', type=float, help="minimum score for a candidate box to be considered as positive in the visualisation", default=0.5) #0.5
# parser.add_argument('-k', '--top_k', type=int, help="if there are a lot of resulting detection across all classes, keep only the top 'k'", default=100) #100
# parser.add_argument('-o', '--output_dir', type=str, help="path to output", default=r"../data/predictions/")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_boxes(path_to_dir, subject, confidence_threshold=0.5):
    path = pjoin(path_to_dir, f"sub-{subject}_preds.json")
    with open(path, "r") as json_file:
        infos = json.load(json_file)
        infos = infos.values()
    
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    
    for det_box_frac, _, det_label, det_score in infos:
        if det_score >= confidence_threshold:
            det_boxes.append(det_box_frac)
            det_labels.append(det_label)
            det_scores.append(det_score)
    
    return torch.FloatTensor(det_boxes), torch.LongTensor(det_labels), torch.FloatTensor(det_scores)
        

def evaluate(path_to_preds ,predict_subset="train", n_classes=1, percentage=1.,confidence_threshold=0.5):
    """

    Args:
        path_to_model: path to model to evaluate
        path_to_preds: path to the predictions folder. Prediction folder should contain a .csv, a .json and a .nii.gz
        per subject
        predict_subset: whether to test on the training set ("train") or the validation set ("validation")
        n_classes: number of classes in the dataset
        percentage: percentage of the dataset to test

    Returns:
        mAP: mean average precision for the dataset

    """
    dataset = ExampleDataset(n_classes=n_classes, percentage=args.percentage,)
    dataset.batch_size = 32
    dataset.setup(stage="predict")
    loader = dataset.predict_train_dataloader() if predict_subset == "train" \
        else dataset.predict_test_dataloader()

    gt_boxes = list()
    gt_labels = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                # Retrieve predictions from the specified path (requires to run predict.py beforehand)
                batch_preds = [retrieve_boxes(path_to_preds, subj,confidence_threshold=confidence_threshold) for subj in batch["subject"]]
            except FileNotFoundError:
                continue
            boxes, labels = batch["seg"] #ground truth

            # separate into different variables and push everything to cuda
            det_boxes_batch = [x.to(device) for x, _, _ in batch_preds]
            det_labels_batch = [y.to(device) for _, y, _ in batch_preds]
            det_scores_batch = [z.to(device) for _, _, z in batch_preds]
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # aggregate results
            gt_boxes.extend(boxes)
            gt_labels.extend(labels)
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)

        gt_difficulties = [torch.BoolTensor([False] * len(x)).to(device) for x in gt_labels]
        # APs, mAP, precisions_per_class = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties)
        print("\n+-+-+- Computing metrics! +-+-+-+")
        metrics_05 = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties, min_overlap=0.5, return_detail=True)
        print("mAP: ", metrics_05["mAP"])
        print("precision: ", metrics_05["precision"])
        print("recall: ", metrics_05["recall"])
        print("f1_score: ", metrics_05["f1_score"])
        print()

        metrics_01 = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties, min_overlap=0.1, return_detail=True)
        print("mAP:       ", metrics_01["mAP"])
        print("precision: ", metrics_01["precision"])
        print("recall:    ", metrics_01["recall"])
        print("f1_score:  ", metrics_01["f1_score"])


    # print("Average precisions: ", APs)
    # print("Mean average precisions: ", mAP)
    # print("Precisions per class & per recall threshold:\n ", precisions_per_class)


if __name__ == "__main__":
    path_to_preds = r"/home/wynen/PycharmProjects/MSLesions3D/data/predictions"

    for ct in [0.1,0.2,0.3,0.4,0.5,]:
        print(f"Confidence threshold set to {ct}")
        evaluate( path_to_preds, percentage=1., confidence_threshold=ct)

    # print("Confidence threshold set to 0.1")
    # evaluate(path_to_preds, percentage=1., confidence_threshold=0.1)

    # print("Confidence threshold set to 0.9")
    # evaluate(path_to_preds, percentage=1., confidence_threshold=0.9)
