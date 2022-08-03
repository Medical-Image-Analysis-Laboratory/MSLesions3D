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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_boxes(path_to_dir, subject):
    path = pjoin(path_to_dir, f"sub-{subject}_preds.json")
    with open(path, "r") as json_file:
        infos = json.load(json_file)
        infos = infos.values()
    
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    
    for det_box_frac, _, det_label, det_score in infos:
        det_boxes.append(det_box_frac)
        det_labels.append(det_label)
        det_scores.append(det_score)
    
    return torch.FloatTensor(det_boxes), torch.LongTensor(det_labels), torch.FloatTensor(det_scores)
        



if __name__ == "__main__":

    path_to_dir = r"C:\Users\Cristina\Desktop\MSLesions3D\data\example\one_class\predictions"
    predict_subset = 'train'
    n_classes = 1
    dataset = ExampleDataset(n_classes = n_classes, percentage = 1., cache=False)
    dataset.batch_size = 16
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
            boxes, labels = batch["seg"]
            
            batch_preds = [retrieve_boxes(path_to_dir, subj) for subj in batch["subject"]]
            
            det_boxes_batch  = [x.to(device) for x,_,_ in batch_preds]
            det_labels_batch = [y.to(device) for _,y,_ in batch_preds]
            det_scores_batch = [z.to(device) for _,_,z in batch_preds]
            
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            gt_boxes.extend(boxes)
            gt_labels.extend(labels)
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            
            
        gt_difficulties = [torch.BoolTensor([False]).to(device)] * len(gt_labels)
        APs, mAP, average_precisions = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties)
