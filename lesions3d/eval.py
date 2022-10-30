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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset_path', type=str, help="path to dataset used for training and validation",
                    default=r'../data/artificial_dataset')
parser.add_argument('-dn', '--dataset_name', type=str, help="name of dataset to use", default=None)
parser.add_argument('-mn', '--model_name', type=str, help="wandb model name", default=None)
parser.add_argument('-p', '--percentage', type=float, help="percentage of the dataset to predict on", default=1.)
parser.add_argument('-c', '--n_classes', type=int, help="number of classes in dataset", default=1)
parser.add_argument('-nw', '--num_workers', type=int, default=8, help="number of workers for the dataset")
parser.add_argument('-ps', '--predict_subset', type=str, help="subset to predict on",
                    choices=['train', 'validation', 'test', 'all'], default=r'train')
parser.add_argument('-sc', '--min_score', type=float,
                    help="minimum score for a candidate box to be considered as positive in the visualisation",
                    default=0.5)
parser.add_argument('-iou', '--min_iou', type=float,
                    help="minimum overlap between the candidate box and a ground truth "
                         "box to be considered as positive", default=0.5)
parser.add_argument('-k', '--top_k', type=int,
                    help="if there are a lot of resulting detection across all classes, keep only the top 'k'",
                    default=100)
parser.add_argument('-pd', '--prediction_dir', type=str, help="path to output", default=r"../data/predictions/")
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


def evaluate(prediction_dir, dataset_path, model_name, dataset_name=None, num_workers=8, predict_subset="train", n_classes=1,
             percentage=1., confidence_threshold=0.5, min_iou=0.5):
    """

    Args:
        path_to_model: path to model to evaluate
        prediction_dir: path to the predictions folder. Prediction folder should contain a .csv, a .json and a .nii.gz
        per subject
        predict_subset: whether to test on the training set ("train") or the validation set ("validation")
        n_classes: number of classes in the dataset
        percentage: percentage of the dataset to test

    Returns:
        mAP: mean average precision for the dataset

    """
    dataset = ExampleDataset(n_classes=n_classes, percentage=percentage, cache=False, num_workers=num_workers,
                             objects="multiple", batch_size=1, data_dir=dataset_path, dataset_name=dataset_name)
    dataset.batch_size = 32
    dataset.setup(stage="predict")
    loader = dataset.predict_train_dataloader() if predict_subset == "train" \
        else dataset.predict_test_dataloader()

    prediction_dir = prediction_dir if dataset_name is None else pjoin(prediction_dir, dataset_name)
    prediction_dir = prediction_dir if model_name is None else pjoin(prediction_dir, model_name)
    prediction_dir = pjoin(prediction_dir, f"{predict_subset}_set")
    prediction_dir = pjoin(prediction_dir, f"min_score_0.0")
    # prediction_dir = pjoin(prediction_dir, f"min_score_{confidence_threshold}")
    if not pexists(prediction_dir):
        raise FileNotFoundError("Prediction directory does not exist: Predictions at min_score=0.0 must be done beforehand.")

    gt_boxes = list()
    gt_labels = list()
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                # Retrieve predictions from the specified path (requires to run predict.py beforehand)
                batch_preds = [retrieve_boxes(prediction_dir, subj, confidence_threshold=confidence_threshold) for subj
                               in batch["subject"]]
            except FileNotFoundError:
                continue
            boxes, labels = batch["seg"]  # ground truth

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
        print("\n+-+-+- Computing metrics! +-+-+-+")
        metrics = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties,
                                   min_overlap=min_iou, return_detail=True)

        print(f"\n\n_________________________AP for IoU = {min_iou} / min score = {confidence_threshold}_________________________\n")
        print("mAP: ", metrics["mAP"])
        print("precision: ", metrics["precision"])
        print("recall: ", metrics["recall"])
        print("f1_score: ", metrics["f1_score"])
        print()

        def convert_tensor(tensor):
            try:
                return tensor.cpu().detach().item()
            except ValueError:
                return tensor.cpu().detach().tolist()

        metrx = {}
        for key, value in metrics.items():
            if type(value) in [int, float, str]:
                metrx[key] = value
            elif type(value) == dict:
                metrx[key] = {k: convert_tensor(v) for k, v in value.items()}
            else:
                metrx[key] = convert_tensor(value)

        print(metrx)
        with open(pjoin(prediction_dir, f"metrics_(min_IoU={min_iou}_min_score={confidence_threshold}).json"), "w") as json_file:
            json.dump(metrx, json_file, indent=4)


if __name__ == "__main__":
    # path_to_preds = r"/home/wynen/PycharmProjects/MSLesions3D/data/predictions"

    # for ct in [0.1,0.2,0.3,0.4,0.5,]:
    #     print(f"Confidence threshold set to {ct}")
    #     evaluate(path_to_preds, percentage=1., confidence_threshold=ct)

    print(f"Confidence threshold set to {args.min_score}")
    evaluate(args.prediction_dir, args.dataset_path, dataset_name=args.dataset_name, model_name=args.model_name,
             num_workers=args.num_workers, predict_subset=args.predict_subset, n_classes=args.n_classes,
             percentage=args.percentage, confidence_threshold=args.min_score, min_iou=args.min_iou)

    # print("Confidence threshold set to 0.9")
    # evaluate(path_to_preds, percentage=1., confidence_threshold=0.9)
