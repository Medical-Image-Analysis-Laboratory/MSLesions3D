# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:31:15 2022

@author: Maxence Wynen
"""

import torch
from datasets import *
from utils import *
from copy import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_metric(predict_subset, modif_lambda=lambda x: x, n_fp_per_image = 0):
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
            if i == 0:
                print("Finally iterating!")
            boxes, labels = batch["seg"]
            
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            
            det_boxes_batch  = [torch.clip(modif_lambda(box),0.,1.) for box in boxes]
            
            dbb = copy(det_boxes_batch)
            
            for k, dboxes in enumerate(det_boxes_batch):
                added_boxes = []
                for j in range(n_fp_per_image):
                    box = torch.FloatTensor([.0,.0,.0,.0,.0,.0]).to(device)
                    box[j%6] = .1 * (j//6) * ((j%6)+1) * (-1.) * (j%2)
                    box = torch.clip(dboxes[0] + box, 0., 1.)
                    added_boxes.append(box)
                    
                dbb[k] = torch.stack([dbb[k].squeeze()] + added_boxes).to(device)
            
            det_boxes_batch = dbb           
            
            det_labels_batch = [torch.LongTensor([1] * (n_fp_per_image + 1)) ] * (len(boxes))
            det_scores_batch = [torch.FloatTensor([1.] * (n_fp_per_image + 1)) ] * (len(boxes))
            
            gt_boxes.extend(boxes)
            gt_labels.extend(labels)       
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            
            
        gt_difficulties = [torch.BoolTensor([False]).to(device)] * len(gt_labels)
        APs, mAP, average_precisions = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties)

    print("mAP:",mAP, APs)
    return average_precisions


if __name__ == "__main__":
    pass
    
    # test_metric("train", lambda x: x + torch.FloatTensor([.01,.01,.01,.01,.01,.01]).to(device))
    
    for i in [1,2,3,5,10,50,100]:
        print("# False positives per image: ", i, end = "   ///  ")
        test_metric("train", n_fp_per_image=i)
    
    # predict_subset = "train" 
    # modif_lambda=lambda x: x
    # n_fp_per_image = 1
    

    
