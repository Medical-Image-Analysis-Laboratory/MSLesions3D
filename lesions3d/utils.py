# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:42:45 2022

@author: Maxence Wynen
"""
import torch
import numpy as np
import copy
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from monai.config import KeysCollection
from monai.transforms import RandCropByPosNegLabeld, FgBgToIndices
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.inverse import InvertibleTransform
from copy import deepcopy
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union, List
from monai.config.type_definitions import NdarrayOrTensor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from monai.data import box_area
from monai.utils.deprecate_utils import deprecated_arg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
# voc_labels = tuple(config["dataset"]["classes"])
voc_labels = tuple(["lesion"])
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080',
                   '#FFFFFF', '#B99E43', '#A4B943', '#7AB943', '#43B969', '#43B993',
                   '#43B9B9', '#4399B9', '#4375B9', '#4358B9', '#4A43B9', '#7A43B9',
                   '#A743B9']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def cxcycz_to_xyz(cxcycz):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, c_z, w, h, z) to boundary 
    coordinates (x_min, y_min, z_min, x_max, y_max, z_max).

    :param cxcycz: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 6)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 6)
    """
    return torch.cat([cxcycz[:, :3] - (cxcycz[:, 3:] / 2),  # x_min, y_min, z_min
                      cxcycz[:, :3] + (cxcycz[:, 3:] / 2)], 1)  # x_max, y_max, z_max


def gcxgcygcz_to_cxcycz(gcxgcygcz, priors_cxcycz):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcygcz: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 6)
    :param priors_cxcycz: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 6)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 6)
    """

    return torch.cat([gcxgcygcz[:, :3] * priors_cxcycz[:, 3:] / 10 + priors_cxcycz[:, :3],  # c_x, c_y, c_z
                      torch.exp(gcxgcygcz[:, 3:] / 5) * priors_cxcycz[:, 3:]], 1)  # w, h, z


def cxcycz_to_gcxgcygcz(cxcycz, priors_cxcycz):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcycz: bounding boxes in center-size coordinates, a tensor of size (n_priors, 6)
    :param priors_cxcycz: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 6)
    :return: encoded bounding boxes, a tensor of size (n_priors, 6)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcycz[:, :3] - priors_cxcycz[:, :3]) / (priors_cxcycz[:, 3:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcycz[:, 3:] / priors_cxcycz[:, 3:]) * 5], 1)  # g_w, g_h


def xyz_to_cxcycz(xy):
    """
    Convert bounding boxes from boundary coordinates 
        (x_min, y_min, z_min, x_max, y_max, z_max) to center-size coordinates 
        (c_x, c_y, c_z, w, h, d).

    :param xyz: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 6)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 6)
    """
    return torch.cat([(xy[:, 3:] + xy[:, :3]) / 2,  # c_x, c_y, c_z
                      xy[:, 3:] - xy[:, :3]], 1)  # w, h, d


def find_intersection3d(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # print("Find intersection")
    # print(set_1.shape)
    # print(set_2.shape)
    # import sys
    # sys.exit()
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :3].unsqueeze(1), set_2[:, :3].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 3:].unsqueeze(1), set_2[:, 3:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1] * intersection_dims[:, :, 2]  # (n1, n2)


def find_jaccard_overlap3d(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 6)
    :param set_2: set 2, a tensor of dimensions (n2, 6)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection3d(set_1, set_2)  # (n1, n2)

    # Find volume of each box in both sets
    volumes_set_1 = (set_1[:, 3] - set_1[:, 0]) * \
                    (set_1[:, 4] - set_1[:, 1]) * \
                    (set_1[:, 5] - set_1[:, 2])  # (n1)
    volumes_set_2 = (set_2[:, 3] - set_2[:, 0]) * \
                    (set_2[:, 4] - set_2[:, 1]) * \
                    (set_2[:, 5] - set_2[:, 2])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = volumes_set_1.unsqueeze(1) + volumes_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def volume(box):
    vol = (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])
    return vol


def compute_metrics_per_class(det_class_images, det_class_boxes, det_class_scores,
                              true_class_images, true_class_boxes, true_class_difficulties, min_overlap):
    """
    Computes some metrics for a specific class

    Args:
        det_class_images: Tensor of size (n_class_detections) containing class detected objects' images
        det_class_boxes: Tensor of size (n_class_detections, 6) containing class detected objects' bounding boxes
        det_class_scores: Tensor of size (n_class_detections, 6) containing class detected objects' scores
        true_class_images: Tensor of size (n_class_objects) containing class true objects' images
        true_class_boxes: Tensor of size (n_class_objects, 6) containing class true objects' boxes
        true_class_difficulties: Tensor of size (n_class_objects) containing class true objects' difficulties (0 or 1)
        min_overlap: (float) minimum overlap a detected box must have with a gt box for it to be considered as true pos

    Returns:
        true_positives: Tensor of size (n_class_detections) containing 1 if detected object at index is a true positive else 0
        false_positives: Tensor of size (n_class_detections) containing 1 if detected object at index is a false positive else 0
        true_class_boxes_detected: Tensor of size (n_class_objects) containing 1 if true box was detected otherwise 0
        true_class_boxes_volumes: Tensor of size (n_class_objects) containing volumes of TP detected boxes (other
            entries are zero)
        det_class_images: Tensor of size (n_class_detections) image to which the detected box at index belongs
            (correspondance with other return values)
        det_class_scores: Tensor of size (n_class_detections) sorted scores for recalculation of mAP and PR curves
    """
    # Keep track of which true objects with this class have already been 'detected'
    # So far, none
    true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)

    n_class_detections = det_class_boxes.size(0)

    # Sort detections in decreasing order of confidence/scores
    det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
    det_class_images = det_class_images[sort_ind]  # (n_class_detections)
    det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

    # In the order of decreasing scores, check if true or false positive
    true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
    false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
    for d in range(n_class_detections):
        this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 6)
        this_image = det_class_images[d]  # (), scalar

        # Find objects in the same image with this class, their difficulties, and whether they have been detected before
        object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
        object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
        # If no such object in this image, then the detection is a false positive
        if object_boxes.size(0) == 0:
            false_positives[d] = 1
            continue

        # Find maximum overlap of this detection with objects in this image of this class
        overlaps = find_jaccard_overlap3d(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
        max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

        # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
        # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
        original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
        # We need 'original_ind' to update 'true_class_boxes_detected'

        # If the maximum overlap is greater than the threshold of 0.5, it's a match
        if max_overlap.item() > min_overlap:
            # If the object it matched with is 'difficult', ignore it
            if object_difficulties[ind] == 0:
                # If this object has already not been detected, it's a true positive
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positives[d] = 1
        # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
        else:
            false_positives[d] = 1
    class_boxes_volumes = torch.FloatTensor([volume(b) for i, b in enumerate(true_class_boxes) if not true_class_difficulties[i]]).to(device)
    # try:
    class_found_boxes_volumes = class_boxes_volumes[true_class_boxes_detected == 1]
    class_not_found_boxes_volumes = class_boxes_volumes[true_class_boxes_detected == 0]
    # except IndexError as e:
    #     print(class_boxes_volumes)
    #     print(true_class_boxes_detected.shape)
    #     raise Exception("Debugging")

    return true_positives, false_positives, true_class_boxes_detected, det_class_scores, class_found_boxes_volumes, class_not_found_boxes_volumes


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, min_overlap=0.5,
                  return_detail=False):
    """
    Calculate the Mean Average Precision (mAP) of detected objects. TODO
    
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: TODO
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(
        true_difficulties)
    # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 6)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 6)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    true_positives_per_class = {}
    false_positives_per_class = {}
    true_boxes_detected_per_class = {}
    found_boxes_volumes_per_class = {}
    not_found_boxes_volumes_per_class = {}
    sorted_scores_per_class = {}
    recalls_per_class = {}
    precisions_per_class = {}
    f1_scores_per_class = {}
    n_easy_class_objects = 0
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 6)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (true_class_difficulties.logical_not()).sum().item()  # ignore difficult objects

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 6)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        metrics = compute_metrics_per_class(det_class_images, det_class_boxes, det_class_scores,
                                            true_class_images, true_class_boxes, true_class_difficulties, min_overlap)
        class_true_positives, class_false_positives, true_class_boxes_detected, det_class_scores_sorted,\
            class_found_boxes_volumes, class_not_found_boxes_volumes = metrics

        true_positives_per_class[c] = class_true_positives
        false_positives_per_class[c] = class_false_positives
        true_boxes_detected_per_class[c] = true_class_boxes_detected
        found_boxes_volumes_per_class[c] = class_found_boxes_volumes
        not_found_boxes_volumes_per_class[c] = class_not_found_boxes_volumes
        sorted_scores_per_class[c] = det_class_scores_sorted

        class_false_negatives = 1 - true_class_boxes_detected
        recalls_per_class[c] = class_true_positives.sum() / (class_true_positives.sum() + class_false_negatives.sum())
        precisions_per_class[c] = class_true_positives.sum() / (class_true_positives.sum() + class_false_positives.sum())
        f1_scores_per_class[c] = (2 * precisions_per_class[c] * recalls_per_class[c]) / (precisions_per_class[c] + recalls_per_class[c])

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(class_true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(class_false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # recalls_per_class = {c: (ctps.sum() / (true_boxes_detected_per_class[c].sum() + ).item()
    #                      for c, ctps in true_positives_per_class.items()}
    # precisions_per_class = {c: (ctps.sum() / (ctps.sum() + false_positives_per_class[c].sum())).item()
    #                          for c, ctps in true_positives_per_class.items()}
    # f1_score_per_class = {c: 2*(prec * recalls_per_class[c])/(prec + recalls_per_class[c])
    #                       for c, prec in true_positives_per_class.items()}


    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    if n_classes == 2:
        try:
            recalls_per_class = recalls_per_class[1]
            precisions_per_class = precisions_per_class[1]
            f1_scores_per_class = f1_scores_per_class[1]
            average_precisions = average_precisions[list(average_precisions.keys())[0]]
            true_boxes_detected_per_class = true_boxes_detected_per_class[1]
            found_boxes_volumes_per_class = found_boxes_volumes_per_class[1]
            not_found_boxes_volumes_per_class = not_found_boxes_volumes_per_class[1]
            true_positives_per_class = true_positives_per_class[1]
            false_positives_per_class = false_positives_per_class[1]
        except KeyError: # no detected objects
            recalls_per_class = 0.
            precisions_per_class = 0.
            f1_scores_per_class = 0.
            average_precisions = 0.
            true_boxes_detected_per_class = torch.zeros(n_easy_class_objects, dtype=torch.uint8).to(device)
            true_positives_per_class = torch.Tensor([]).to(device)
            false_positives_per_class = torch.Tensor([]).to(device)
            found_boxes_volumes_per_class = torch.Tensor([]).to(device)
            true_boxes_volumes = torch.FloatTensor([volume(b) for b in true_boxes]).to(device)
            not_found_boxes_volumes_per_class = true_boxes_volumes

    if not return_detail:
        return average_precisions, mean_average_precision
    else:
        return {"APs":                                  average_precisions,
                "mAP":                                  mean_average_precision,
                "precision":                            precisions_per_class,
                "recall":                               recalls_per_class,
                "f1_score" :                            f1_scores_per_class,
                "sorted_det_scores" :                   sorted_scores_per_class,
                "TP" :                                  true_positives_per_class,
                "FP" :                                  false_positives_per_class,
                "n_true_boxes":                         true_boxes_detected_per_class.size(0),
                "found_boxes_volumes_per_class":        found_boxes_volumes_per_class,
                "not_found_boxes_volumes_per_class":    not_found_boxes_volumes_per_class,
                }

class BoundingBoxesGeneratord(MapTransform, InvertibleTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False,
                 segmentation_mode: str = "instances",
                 thresholds: list = None,
                 classes=None,
                 n_classes: int = None) -> None:
        super().__init__(keys, allow_missing_keys)
        self.segmentation_mode = segmentation_mode
        self.thresholds = thresholds
        if n_classes is not None and not classes:
            classes = list(range(1, n_classes + 1))
        self.classes = classes
        if not n_classes and classes:
            n_classes = len(classes)

        assert segmentation_mode in ["instances", "binary", "classes"]
        assert segmentation_mode != "instances" or thresholds
        assert segmentation_mode != "binary" or (not classes and not n_classes) or n_classes == 1
        assert segmentation_mode != "classes" or type(classes) in [list, dict]

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            self.push_transform(d, key)
            gt_bboxes, gt_labels = self.converter(d[key])
            d["boxes"] = gt_bboxes
            d["labels"] = gt_labels

        return d

    def inverse(self, data: dict) -> dict:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # Do nothing
            self.pop_transform(d, key)
        return d

    def converter(self, seg):
        seg = np.squeeze(seg)
        image_size = seg.shape

        if self.segmentation_mode == "instances":
            gt_bboxes, gt_labels = self._from_instances(seg)

        elif self.segmentation_mode == "binary":
            connected_components, n_objects = label(seg)
            thresholds = [(1, np.inf)]
            gt_bboxes, gt_labels = self._from_instances(connected_components, thresholds)

        elif self.segmentation_mode == "classes":
            if max(self.classes) < np.max(seg):
                print("Warning")
                warnings.warn(
                    "Number of classes in segmentation does not correspond to number of classes given as input.")

            thresholds = []
            seg_instanced = copy.copy(seg)

            for c in self.classes:
                class_seg = copy.copy(seg)
                class_seg[class_seg != c] = 0
                class_connected_components, n_objects = label(class_seg)
                seg_instanced = np.where(class_seg == c,
                                         class_connected_components + (c * 1000),
                                         seg_instanced)
                thresholds.append((c * 1000, (c + 1) * 1000))

            gt_bboxes, gt_labels = self._from_instances(seg_instanced, thresholds)
        else:
            raise ValueError(f"Unknown segmentation_mode={self.segmentation_mode}")

        if len(gt_bboxes) != 0:
            gt_bboxes = torch.FloatTensor(gt_bboxes) / torch.FloatTensor(image_size * 2)
            gt_labels = torch.LongTensor(gt_labels)

            # Filter areas = 0
            areas = box_area(gt_bboxes)
            zero_area_indices = torch.argwhere(areas == 0.0).flatten()
            for i, idx in enumerate(zero_area_indices):
                idx = int(idx) - i
                gt_bboxes = torch.cat((gt_bboxes[:idx, :], gt_bboxes[idx + 1:, :]))
                gt_labels = torch.cat((gt_labels[:idx], gt_labels[idx + 1:]))

            return gt_bboxes, gt_labels
        else:
            return torch.FloatTensor(gt_bboxes), torch.LongTensor(gt_labels)

    def _from_instances(self, seg, thresholds=None):
        gt_bboxes = []
        gt_labels = []

        seg = np.squeeze(seg)

        thresholds = self.thresholds if not thresholds else thresholds

        labels = np.delete(np.unique(seg), 0)
        indices = np.array([np.where(seg == l) for l in labels if l != 0], dtype=object)
        for c, (min_value, max_value) in enumerate(thresholds):
            class_indices = np.where((labels >= min_value) & (labels < max_value))[0]
            class_indices = indices[class_indices]
            # 3 Dimensions
            if len(seg.shape) == 3:
                class_bboxes = [[min(x), min(y), min(z), max(x), max(y), max(z)] \
                                for x, y, z in class_indices]
            # 2 Dimensions
            elif len(seg.shape) == 2:
                class_bboxes = [[min(x), min(y), max(x), max(y)] \
                                for x, y in class_indices]
            else:
                raise NotImplementedError(
                    f"Unknown shape for segmentation. Must have either 2 or 3 dimensions but got {len(seg.shape)}.")

            gt_bboxes += class_bboxes
            gt_labels += [c + 1] * len(class_bboxes)

        return gt_bboxes, gt_labels


def make_segmentation_from_bboxes(bboxes: NdarrayOrTensor, labels: NdarrayOrTensor, shape: tuple,
                                  return_type: str = "torch", return_batch_first=True):
    """
    boxes: NdarrayOrTensor of shape (batch_dim, n_boxes, 6) or (n_boxes, 6)
    labels: NdarrayOrTensor of shape (batch_dim, n_boxes) or (n_boxes)
    shape: tuple of image dimensions

    return: (pred_boxes: torch.FloatTensor of shape (batch_size, n_boxes, shape),
            pred_labels: torch.FloatTensor of shape (batch_size, n_boxes, shape))
            where pred_boxes is an torch.FloatTensor of shape (batch_size, n_boxes, shape) with the edges of
            each box in boxes for each image instance have a different label (box_1 = 1, box_2 = 2 etc)
            and pred_labels is a torch.FloatTensor of shape (batch_size, n_boxes, shape) where the edges of
            each box in boxes for each image instance have the label of the corresponding class label found
            in labels
    """

    def assertions(bboxes, labels):
        assert bboxes.shape[0] == labels.shape[0]
        assert len(bboxes.shape) in (2, 3)

    if type(bboxes) == list:
        for bbox, blabels in zip(bboxes, labels):
            assertions(bbox, blabels)
    else:
        assertions(bboxes, labels)
    assert return_type in ("torch", "numpy")

    # Put in batch dimension-first format (batch_dim, n_boxes, 6)
    if len(bboxes.shape) == 2:
        bboxes = torch.unsqueeze(torch.FloatTensor(bboxes), dim=0)  # (batch_dim, n_boxes, 6)
        labels = torch.unsqueeze(torch.Tensor(labels), dim=0)  # (batch_dim, n_boxes)
    elif not return_batch_first:
        warnings.warn("return_batch_first is False but a batch of boxes was given. return_batch_first set to True")
        return_batch_first = True

    batch_pred_boxes = list()
    batch_pred_labels = list()

    for i, image_boxes in enumerate(bboxes):

        pred_boxes = torch.squeeze(torch.zeros(shape))  # (shape)
        pred_labels = torch.squeeze(torch.zeros(shape))  # (shape)

        for j, det_box in enumerate(image_boxes):
            if det_box.shape == torch.Size([0]):
                continue
            det_label = int(labels[i][j])

            if det_label == 0:  # Background
                continue

            det_box = torch.clip(det_box, 0, 1)  # Make sure the box does not exceed the image edges
            det_box *= torch.Tensor(shape * 2)  # Put fractional boxes form into actual size form
            det_box = det_box.numpy().astype(int).tolist()  # Convert to integer for slicing
            x_min, y_min, z_min, x_max, y_max, z_max = det_box

            # Make sure the box does not exceed the image edges
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            z_min = max(z_min, 0)
            x_max = min(x_max, shape[0] - 1)
            y_max = min(y_max, shape[1] - 1)
            z_max = min(z_max, shape[2] - 1)

            # Classes
            pred_labels[x_min, y_min:y_max, z_min:z_max] = det_label
            pred_labels[x_max, y_min:y_max, z_min:z_max] = det_label

            pred_labels[x_min:x_max, y_min, z_min:z_max] = det_label
            pred_labels[x_min:x_max, y_max, z_min:z_max] = det_label

            pred_labels[x_min:x_max, y_min:y_max, z_min] = det_label
            pred_labels[x_min:x_max, y_min:y_max, z_max] = det_label

            # Instances
            pred_boxes[x_min, y_min:y_max, z_min:z_max] = j + 1
            pred_boxes[x_max, y_min:y_max, z_min:z_max] = j + 1

            pred_boxes[x_min:x_max, y_min, z_min:z_max] = j + 1
            pred_boxes[x_min:x_max, y_max, z_min:z_max] = j + 1

            pred_boxes[x_min:x_max, y_min:y_max, z_min] = j + 1
            pred_boxes[x_min:x_max, y_min:y_max, z_max] = j + 1

        pred_boxes = torch.unsqueeze(pred_boxes, dim=0)  # (1, shape)
        pred_labels = torch.unsqueeze(pred_labels, dim=0)  # (1, shape)

        batch_pred_boxes.append(pred_boxes)
        batch_pred_labels.append(pred_labels)

    batch_pred_boxes = torch.concat(batch_pred_boxes, dim=0)
    batch_pred_labels = torch.concat(batch_pred_labels, dim=0)

    if return_batch_first:
        batch_pred_boxes = torch.unsqueeze(batch_pred_boxes, dim=0)
        batch_pred_labels = torch.unsqueeze(batch_pred_labels, dim=0)

    if return_type == "numpy":
        batch_pred_boxes = batch_pred_boxes.cpu().numpy()
        batch_pred_labels = batch_pred_labels.cpu().numpy()

    return batch_pred_boxes, batch_pred_labels


def show_image(image, slice, dim, title=""):
    img = torch.FloatTensor(image)
    img = np.squeeze(img.cpu().numpy())
    assert len(img.shape) == 3 or len(img.shape) == 4

    slice = int(slice * img.shape[dim])

    def s(img, title):
        if dim == 0:
            plt.imshow(img[slice, :, :], cmap="gray")
        if dim == 1:
            plt.imshow(img[:, slice, :], cmap="gray")
        if dim == 2:
            plt.imshow(img[:, :, slice], cmap="gray")
        plt.title(title + f" (shape: {img.shape[dim]}, slice {slice}; dim {dim})")
        plt.colorbar()
        plt.show()

    if type(title) == list:
        for i, ig in enumerate(img.tolist()):
            s(ig, title[i])
    else:
        s(img, title)


def show_multiple_images(list_of_images, slice, dim, titles=[], plt_title=""):
    titles = [""] * len(list_of_images) if titles is None else titles

    def s(fig, ax, img, title):
        if dim == 0:
            im = ax.imshow(img[slice, :, :], cmap="gray")
        if dim == 1:
            im = ax.imshow(img[:, slice, :], cmap="gray")
        if dim == 2:
            im = ax.imshow(img[:, :, slice], cmap="gray")
        else:
            raise AssertionError
        ax.set_title(title + f" (slice {slice}; dim {dim})")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    if len(list_of_images) == 2:
        axes = (1, 2)
    else:
        raise NotImplementedError()

    fig, axes = plt.subplots(*axes, figsize=(8, 4))
    fig.suptitle(plt_title)

    slice_computed = False
    for i, image in enumerate(list_of_images):
        img = torch.FloatTensor(image)
        img = np.squeeze(img.cpu().numpy())
        assert len(img.shape) == 3 or len(img.shape) == 4

        if not slice_computed:
            slice = int(slice * img.shape[dim])
            slice_computed = True

        s(fig, axes[i], img, titles[i])

    fig.tight_layout()

    plt.show()

class CustomRandCropByPosNegLabeld(RandCropByPosNegLabeld):
    """
    Custom version of the dictionary-based version of :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding metadata.
    And will return a list of dictionaries for all the cropped images.

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center
    to ensure the valid crop ROI.

    The main difference with RandCropByPosNegLabeld is that this version first categorizes the foreground
    and background voxels before sampling the crop center.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, label_key, spatial_size, pos, neg, num_samples, image_key, image_threshold,
            None, None, meta_keys, meta_key_postfix, allow_smaller, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> List[Dict[Hashable, torch.Tensor]]:
        find_fg_bg_indices = FgBgToIndices(image_threshold=-0.1)
        self.fg_indices_key, self.bg_indices_key = find_fg_bg_indices(data["seg"])

        return super.__call__(data)



class ShowImage(MapTransform, InvertibleTransform):
    def __init__(self,
                 keys: KeysCollection,
                 dim: int,
                 slice: float = 0.5,
                 msg: str = "",
                 grid: bool = False,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.slice = slice
        self.dim = dim
        self.msg = msg
        self.grid = grid

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if not self.grid:
            for key in self.key_iterator(d):
                show_image(d[key], self.slice, self.dim, f"({self.msg}) Subject {d['subject']}")
        else:
            show_multiple_images([d[key] for key in self.key_iterator(d)], self.slice,
                                 self.dim, titles=[key for key in self.key_iterator(d)],
                                 plt_title=f"({self.msg}) Subject {d['subject']}")

        return d

    def inverse(self, data: dict) -> dict:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            show_image(d[key], self.slice, self.dim, f"(Inverse)({self.msg}) Subject {d['subject']}")
        return d


class Printer(Callable):
    def __init__(self, string):
        self.string = string

    def __call__(self, arg):
        if type(arg) == str:
            print(self.string, arg)
        elif type(arg) == torch.Tensor or type(arg) == torch.FloatTensor or type(arg) == np.ndarray:
            print(self.string, "(Shape =", arg.shape, ")")
        else:
            print(self.string, f"({type(arg)})")
        return arg
