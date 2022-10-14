# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:39:19 2022

@author: Maxence Wynen
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet import conv_bn, Block
from math import sqrt
from utils import *
import wandb
import os
from os.path import join as pjoin
from os.path import exists as pexists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASPECT_RATIOS = {3:[1.], 5: [1.], 7: [1]}

SCALES = {1: 0.025,
          3: 0.05,
          5: 0.075,
          7: 0.1,
          11: 0.125,
          13: 0.15
          }# EXAMPLES
# SCALES = {3:0.01,
#            5: 0.02,
#             13: 0.075, }

config = [32,
          # channel, n, stride
          [64, 1, (2, 2, 2)],
          [128, 2, (2, 2, 2)],
          [256, 2, (2, 2, 2)],
          [512, 6, (2, 2, 2)],
          [1024, 2, (1, 1, 1)],
          ]

# def get_feature_map_dimensions(input_size):
#     FEATURE_MAP_DIMENSIONS = {}
#
#     cube = input_size[0] == input_size[1] == input_size[2]
#     first_stride = (1,2,2)first_stride = (1,2,2) if not cube else (2,2,2) if not cube else (2,2,2)
#
#     cfg = config[1:]
#     # channel, n_repeat, stride
#     i = 0
#
#     last_feature_map_dim = input_size
#     FEATURE_MAP_DIMENSIONS[i+1] = (last_feature_map_dim[0] // first_stride[0],
#                                      last_feature_map_dim[1] // first_stride[1],
#                                      last_feature_map_dim[2] // first_stride[2],)
#     for _, (c, n, s) in enumerate(cfg):
#         for j in range(n):
#             if j == 0:
#                 FEATURE_MAP_DIMENSIONS[i+1] = (last_feature_map_dim[0] // s[0],) * 3
#                 last_feature_map_dim = FEATURE_MAP_DIMENSIONS[i+1]
#             else:
#                 FEATURE_MAP_DIMENSIONS[i+1] = last_feature_map_dim
#
#             i += 1
#     return FEATURE_MAP_DIMENSIONS






FEATURE_MAP_DIMENSIONS_64 = {0: (32, 32, 32),
                              1: (16, 16, 16),
                              2: (8, 8, 8),
                              3: (8, 8, 8),
                              4: (4, 4, 4),
                              5: (4, 4, 4),
                              6: (2, 2, 2),
                              7: (2, 2, 2),
}

FEATURE_MAP_DIMENSIONS_250 = {0: (250, 300, 300), # FALSE????
                              1: (125, 75, 75),
                              2: (63, 38, 38),
                              3: (63, 38, 38),
                              4: (32, 19, 19),
                              5: (32, 19, 19),
                              6: (16, 10, 10),
                              7: (16, 10, 10),
                              8: (16, 10, 10),
                              9: (16, 10, 10),
                              10: (16, 10, 10),
                              11: (16, 10, 10),
                              12: (16, 10, 10),
                              13: (16, 10, 10),
                              }



features_n_channels = [[cf[0]] * cf[1] for cf in config[1:]]
features_n_channels = [config[0]] + [item for sublist in features_n_channels for item in sublist]


class MobileNetBase(nn.Module):
    def __init__(self, in_channels=1, width_mult=1., cube = False):
        super(MobileNetBase, self).__init__()

        input_channel = config[0]
        input_channel = int(input_channel * width_mult)

        cfg = config[1:]
        first_stride = (1,2,2) if not cube else (2,2,2)
        self.features = [conv_bn(in_channels, input_channel, first_stride)]

        # building inverted residual blocks
        # channel, n_repeat, stride
        for c, n, s in cfg:
            if len(self.features) - 1 == max(ASPECT_RATIOS.keys()): # basically allows to truncate the base network
                break                                               # when you don't need the rest of it
            output_channel = int(c * width_mult)
            for i in range(n):
                if len(self.features) -1 == max(ASPECT_RATIOS.keys()):
                    break
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

    def init(self):
        for c in self.children():
            if isinstance(c, nn.Conv3d):
                nn.init.kaiming_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, image):
        out = image
        out_n_features = list(ASPECT_RATIOS.keys())
        out_features = {}
        for i, feat in enumerate(self.features):
            out = feat(out)
            # print(f"Block {i} output  --  feature shape: {out.shape}")
            if i in out_n_features:
                out_features[i] = out
                if out.isnan().sum() > 0:
                    # breakpoint()
                    print("Yesssss this NaN error again in the base network")
                    raise Exception("Yesssss this NaN error again in the base network")

        return out_features


class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes, width_mult):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {feat: len(ASPECT_RATIOS[feat]) + 1 for feat in ASPECT_RATIOS}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        self.loc_convs = []  # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.cl_convs = []  # Class prediction convolutions (predict classes in localization boxes)
        for f in ASPECT_RATIOS:
            f_n_channels = int(features_n_channels[f] * width_mult)
            self.loc_convs.append(nn.Conv3d(f_n_channels, n_boxes[f] * 6, kernel_size=3, padding=1))
            self.cl_convs.append(nn.Conv3d(f_n_channels, n_boxes[f] * n_classes, kernel_size=3, padding=1))

        self.loc_convs = nn.ModuleList(self.loc_convs)
        self.cl_convs = nn.ModuleList(self.cl_convs)

    def init(self):
        for c in self.children():
            if isinstance(c, nn.Conv3d):
                nn.init.kaiming_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, feats):
        batch_size = feats[min(feats.keys())].size(0)
        feat_keys = list(feats.keys())

        l_convs_out = []
        c_convs_out = []
        for i, (loc_conv, cl_conv) in enumerate(zip(self.loc_convs, self.cl_convs)):
            key = feat_keys[i]
            # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
            l_conv = loc_conv(feats[key])
            l_conv = l_conv.permute(0, 2, 3, 4, 1).contiguous()  # to match prior-box order (after .view())
            # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
            l_conv = l_conv.view(batch_size, -1, 6)
            l_convs_out.append(l_conv)

            # Predict classes in localization boxes
            c_conv = cl_conv(feats[key])
            c_conv = c_conv.permute(0, 2, 3, 4, 1).contiguous()
            c_conv = c_conv.view(batch_size, -1, self.n_classes)
            c_convs_out.append(c_conv)

        # A total of xxx boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat(l_convs_out, dim=1)
        classes_scores = torch.cat(c_convs_out, dim=1)

        return locs, classes_scores

class PredictionConvolutionsMobileNet(nn.Module):
    #TODO
    def __init__(self, n_classes, width_mult):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {feat: len(ASPECT_RATIOS[feat]) + 1 for feat in ASPECT_RATIOS}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        self.loc_convs = []  # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.cl_convs = []  # Class prediction convolutions (predict classes in localization boxes)
        for f in ASPECT_RATIOS:
            f_n_channels = int(features_n_channels[f] * width_mult)
            self.loc_convs.append(nn.Conv3d(f_n_channels, n_boxes[f] * 6, kernel_size=3, padding=1))
            self.cl_convs.append(nn.Conv3d(f_n_channels, n_boxes[f] * n_classes, kernel_size=3, padding=1))

        self.loc_convs = nn.ModuleList(self.loc_convs)
        self.cl_convs = nn.ModuleList(self.cl_convs)

    def init(self):
        for c in self.children():
            if isinstance(c, nn.Conv3d):
                nn.init.kaiming_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, feats):
        batch_size = feats[min(feats.keys())].size(0)
        feat_keys = list(feats.keys())

        l_convs_out = []
        c_convs_out = []
        for i, (loc_conv, cl_conv) in enumerate(zip(self.loc_convs, self.cl_convs)):
            key = feat_keys[i]
            # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
            l_conv = loc_conv(feats[key])
            l_conv = l_conv.permute(0, 2, 3, 4, 1).contiguous()  # to match prior-box order (after .view())
            # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
            l_conv = l_conv.view(batch_size, -1, 6)
            l_convs_out.append(l_conv)

            # Predict classes in localization boxes
            c_conv = cl_conv(feats[key])
            c_conv = c_conv.permute(0, 2, 3, 4, 1).contiguous()
            c_conv = c_conv.view(batch_size, -1, self.n_classes)
            c_convs_out.append(c_conv)

        # A total of xxx boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat(l_convs_out, dim=1)
        classes_scores = torch.cat(c_convs_out, dim=1)

        return locs, classes_scores


class LSSD3D(pl.LightningModule):
    """
    The SSD 3D network - encapsulates the base MobileNet network and the prediction convolutions.
    """

    def __init__(self,
                 n_classes,
                 input_channels=3,
                 input_size=(250,300,300),
                 threshold=0.5,
                 alpha=1.,
                 lr=1.3e-5,
                 width_mult=1.,
                 min_score=0.5,
                 max_overlap=0.5, # for box matching
                 min_overlap=0.5, # for evaluation metrics
                 top_k=100,
                 scheduler="CosineAnnealingLR",
                 use_wandb=False,
                 batch_size=8,
                 compute_metric_every_n_epochs=1,
                 comments="",
                 ASPECT_RATIOS=ASPECT_RATIOS,
                 SCALES=SCALES):
        super(LSSD3D, self).__init__()
        print("Init LSSD3D")

        # self.save_hyperparameters(ignore=["timesteps"])
        self.save_hyperparameters()

        cube = input_size[0] == input_size[1] == input_size[2]
        self.input_size = input_size

        self.n_classes = n_classes
        self.base = MobileNetBase(in_channels=input_channels, width_mult=width_mult, cube=cube)
        self.pred_convs = PredictionConvolutions(n_classes, width_mult=width_mult)
        self.lr = lr
        self.min_score = min_score
        self.max_overlap = max_overlap
        self.min_overlap = min_overlap
        self.top_k = top_k
        self.scheduler = scheduler
        self.input_channels = input_channels
        self.use_wandb = use_wandb
        self.batch_size = batch_size
        self.compute_metric_every_n_epochs = compute_metric_every_n_epochs
        self.comments = comments
        self.ASPECT_RATIOS = ASPECT_RATIOS
        self.SCALES = SCALES
        self.config = config

        # Since lower level features have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        n_channel_rescale = int(features_n_channels[min(ASPECT_RATIOS.keys())] * width_mult)
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, n_channel_rescale, 1, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcycz = self.create_prior_boxes()

        self.loss_fn = MultiBoxLoss(self.priors_cxcycz, threshold=threshold, alpha=alpha)

    def forward(self, image):
        feats = self.base(image)

        # Uncommenting this part gives you NaN values
        # norm = feats[min(feats.keys())].pow(2).sum(dim=1, keepdim=True).sqrt()
        # feats[min(feats.keys())] = feats[min(feats.keys())] / (norm + 1e-6)
        # feats[min(feats.keys())] = feats[min(feats.keys())] * self.rescale_factors

        locs, classes_scores = self.pred_convs(feats)

        if classes_scores.isnan().sum() > 0 :
            raise Exception("Oh no not this NaN error again... (forward SSD), CLASSES_SCORES is nan!")
        if locs.isnan().sum() > 0 :
            raise Exception("Oh no not this NaN error again... (forward SSD), LOCS is nan!")

        return locs, classes_scores

    def create_prior_boxes(self, per_feature_map=False):
        """
        Create the xxx prior (default) boxes for the SSD 3D

        :return: prior boxes in center-size coordinates, a tensor of dimensions (xxx, 6)
        """
        features = list(ASPECT_RATIOS.keys())
        fmd = FEATURE_MAP_DIMENSIONS_250 if self.input_size[0] == 250 else FEATURE_MAP_DIMENSIONS_64
        fmap_dims = {feat: fmd[feat] for feat in features}
        obj_scales = {feat: SCALES[feat] for feat in features}
        aspect_ratios = {feat: ASPECT_RATIOS[feat] for feat in features}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []
        prior_boxes_per_feature_map = {}
        for l, fmap in enumerate(fmaps):
            prior_boxes_per_feature_map[fmap] = list()
            for i in range(fmap_dims[fmap][0]):
                for j in range(fmap_dims[fmap][1]):
                    for k in range(fmap_dims[fmap][2]):
                        cz = (k + 0.5) / fmap_dims[fmap][2]
                        cx = (j + 0.5) / fmap_dims[fmap][1]
                        cy = (i + 0.5) / fmap_dims[fmap][0]

                        for ratio in aspect_ratios[fmap]: # Commented because we only keep an aspect ratio=1
                            prior_boxes.append([cx, cy, cz, obj_scales[fmap], obj_scales[fmap], obj_scales[fmap]])
                            prior_boxes_per_feature_map[fmap].append([cx, cy, cz, obj_scales[fmap],
                                                                      obj_scales[fmap], obj_scales[fmap]])

                            if ratio == 1.:
                                # Add a slightly bigger prior box
                                try:
                                    additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[l + 1]])
                                except IndexError as e:
                                    additional_scale = 1.

                                prior_boxes.append([cx, cy, cz, additional_scale, additional_scale, additional_scale])
                                prior_boxes_per_feature_map[fmap].append([cx, cy, cz, additional_scale,
                                                                          additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)

        if not per_feature_map:
            return prior_boxes
        else:
            return prior_boxes_per_feature_map

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the xxx locations and class scores (output of ths SSD300) to detect objects.
    
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
    
        :param predicted_locs: predicted locations/boxes w.r.t the xxx prior boxes, a tensor of dimensions (N, xxx, 6)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, xxx, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """

        # print("Confidence threshold: ", min_score)

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcycz.size(0)

        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, xxx, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcycz_to_xyz(gcxgcygcz_to_cxcycz(
                predicted_locs[i], self.priors_cxcycz))  # (xxx, 6), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # max_scores, best_label = predicted_scores[i].max(dim=1)  # (xxx)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (xxx)

                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                # print(f"\n\nN boxes: {n_priors} ; Min score: {min_score} ; n_above_min_score: {n_above_min_score}")
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= xxx
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 6)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 6)

                # print("\nClass Scores shape:", class_scores.shape)
                n_above_min_score = min(10 * top_k, n_above_min_score)
                class_scores = class_scores[:n_above_min_score]
                class_decoded_locs = class_decoded_locs[:n_above_min_score]
                # print("\nClass Scores shape after removing:", class_scores.shape)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap3d(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score)).bool().to(device)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    # suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress = suppress | (overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(
                    torch.LongTensor(
                        (~suppress).sum().item() * [c]).to(device)
                )
                image_scores.append(class_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 0., 1., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 6)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 6)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size        

    def init(self):
        self.base.init()
        self.pred_convs.init()

    def training_step(self, batch):

        # Get images, boxes and labels from batch
        images, gt_boxes, gt_labels = batch["img"], batch['boxes'], batch["labels"]
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]

        # Forward prop.
        predicted_locs, predicted_scores = self(images)  # (N, xxx, 6), (N, xxx, n_classes)

        if predicted_scores.isnan().sum() > 0:
            raise Exception("Oh no this NaN error again")  # If this error occurs, try decreasing the learning rate

        for i, subj_boxes in enumerate(gt_boxes):
            for axis in (0, 1, 2):
                negatives = (subj_boxes[:, axis + 3] < subj_boxes[:, axis]).sum()
                zeros = (subj_boxes[:, axis + 3] == subj_boxes[:, axis]).sum()
                if negatives > 0:
                    warnings.warn(f"Given boxes has invalid values (subject {batch['subject'][i]}). The box size must "
                                  f"be non-negative but got {int(negatives)} boxes with negative sizes.")
                if zeros > 0:
                    warnings.warn(f"Given boxes has invalid values (subject {batch['subject'][i]}). The box size must "
                                  f"be non-zero but got {int(zeros)} boxes with size of zero.")

        # Loss
        conf_loss, loc_loss = self.loss_fn(predicted_locs, predicted_scores, gt_boxes, gt_labels)  # scalars
        loss = conf_loss + self.loss_fn.alpha * loc_loss  # scalar

        logs = {"train_total_loss": loss, "train_conf_loss": conf_loss, "train_loc_loss": loc_loss}

        # Compute mAP every n epochs
        if self.current_epoch % (self.compute_metric_every_n_epochs * 2) == 0:

            det_boxes, det_labels, det_scores = self.detect_objects(predicted_locs, predicted_scores,
                                                                    self.min_score, self.max_overlap, self.top_k)
            with torch.no_grad():
                compute_mAP = [1 if len(predicted_locs_img) > 500 else 0 for predicted_locs_img in predicted_locs]
                compute_mAP = sum(compute_mAP)
                if compute_mAP != 0:
                    gt_difficulties = [torch.BoolTensor([False] * lbls.size(0)).to(device) for lbls in gt_labels]
                    # Calculate mAP at min_IoU of 0.1
                    metrics_10 = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties, min_overlap=0.1, return_detail=True)
                    # Calculate mAP at min_IoU of 0.5
                    metrics_50 = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficulties, min_overlap=0.5, return_detail=True)
                else:
                    raise NotImplementedError

            logs["metrics_10"] = metrics_10
            logs["metrics_50"] = metrics_50

        # Log the different losses
        log_fn = self.log if self.use_wandb else self.logger.experiment.add_scalar
        kwargs = [] if self.use_wandb else [self.global_step]
        self.log('total_loss/training', loss.item(), *kwargs)
        self.log('confidence_loss/training', conf_loss.item(), *kwargs)
        self.log('localization_loss/training', loc_loss.item(), *kwargs)


        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        return {'loss': loss, "log": logs}

    def validation_step(self, batch, batch_idx):

        images, [gt_boxes, gt_labels] = batch["img"], batch['seg']

        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]

        # Forward prop.
        predicted_locs, predicted_scores = self(images)  # (N, xxx, 6), (N, xxx, n_classes)

        for i, subj_boxes in enumerate(gt_boxes):
            for axis in (0, 1, 2):
                negatives = (subj_boxes[:, axis + 3] < subj_boxes[:, axis]).sum()
                zeros = (subj_boxes[:, axis + 3] == subj_boxes[:, axis]).sum()
                if negatives > 0:
                    warnings.warn(f"Given boxes has invalid values (subject {batch['subject'][i]}). The box size must "
                                  f"be non-negative but got {int(negatives)} boxes with negative sizes.")
                if zeros > 0:
                    warnings.warn(f"Given boxes has invalid values (subject {batch['subject'][i]}). The box size must "
                                  f"be non-zero but got {int(zeros)} boxes with size of zero.")

        # Loss
        conf_loss, loc_loss = self.loss_fn(predicted_locs, predicted_scores, gt_boxes, gt_labels)  # scalar

        loss = conf_loss + self.loss_fn.alpha * loc_loss

        logs = {"val_total_loss": loss, "val_conf_loss": conf_loss, "val_loc_loss": loc_loss}

        # Compute mAP every n epochs
        if self.current_epoch % self.compute_metric_every_n_epochs == 0:

            det_boxes, det_labels, det_scores = self.detect_objects(predicted_locs, predicted_scores,
                                                                    self.min_score, self.max_overlap, self.top_k)
            with torch.no_grad():
                gt_difficulties = [torch.BoolTensor([False] * lbls.size(0)).to(device) for lbls in gt_labels]
                compute_mAP = [1 if len(predicted_locs_img) > 500 else 0 for predicted_locs_img in predicted_locs]
                compute_mAP = sum(compute_mAP)
                if compute_mAP != 0:
                    gt_difficulties = [torch.BoolTensor([False] * lbls.size(0)).to(device) for lbls in gt_labels]
                    # Calculate mAP at min_IoU of 0.1
                    metrics_10 = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels,
                                                      gt_difficulties, min_overlap=0.1, return_detail=True)
                    # Calculate mAP at min_IoU of 0.5
                    metrics_50 = calculate_mAP(det_boxes, det_labels, det_scores, gt_boxes, gt_labels,
                                                      gt_difficulties, min_overlap=0.5, return_detail=True)
                    metrics_50["mAP"] = torch.FloatTensor([metrics_50["mAP"]])
                else:
                    raise NotImplementedError

            logs["metrics_10"] = metrics_10
            logs["metrics_50"] = metrics_50

        return {'loss': loss, "log": logs}

    def validation_epoch_end(self, outputs):
        # Compute mean losses over all the epoch's batches
        avg_loss = torch.stack([x["log"]["val_total_loss"] for x in outputs]).mean()
        avg_conf_loss = torch.stack([x["log"]["val_conf_loss"] for x in outputs]).mean()
        avg_loc_loss = torch.stack([x["log"]["val_loc_loss"] for x in outputs]).mean()

        # Log the learning rate to keep track of it
        self.log("hp_metric/lr", self.lr)

        # print(f"Logging (validation epoch {self.current_epoch})")#, round(avg_loss.item(),3), round(avg_conf_loss.item(),3), round(avg_loc_loss.item(),3))

        # Log the different losses
        log_fn = self.log if self.use_wandb else self.logger.experiment.add_scalar
        kwargs = [] if self.use_wandb else [self.global_step]
        log_fn('total_loss/validation', avg_loss, *kwargs)
        log_fn('confidence_loss/validation', avg_conf_loss, *kwargs)
        log_fn('localization_loss/validation', avg_loc_loss, *kwargs)

        # Log the mAP every n epochs
        if self.current_epoch % self.compute_metric_every_n_epochs == 0:
            avg_mAP_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["mAP"]]) for x in outputs]).mean()
            avg_precision_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["precision"]])
                                                for x in outputs]).mean() # change this if n_classes > 2
            avg_recall_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["recall"]])
                                         for x in outputs]).mean() # change this if n_classes > 2
            avg_f1_score_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["f1_score"]]) for x in outputs]).mean() # change this if n_classes > 2

            avg_mAP_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["mAP"]]) for x in outputs]).mean()
            avg_precision_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["precision"]]) for x in outputs]).mean() # change this if n_classes > 2
            avg_recall_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["recall"]]) for x in outputs]).mean() # change this if n_classes > 2
            avg_f1_score_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["f1_score"]]) for x in outputs]).mean() # change this if n_classes > 2

            # Logging
            log_fn = self.log if self.use_wandb else self.logger.experiment.add_scalar

            log_fn('mAP/validation_IoU_0.1',         avg_mAP_10,        *kwargs)
            log_fn('precision/validation_IoU_0.1',   avg_precision_10,  *kwargs)
            log_fn('recall/validation_IoU_0.1',      avg_recall_10,     *kwargs)
            log_fn('f1_score/validation_IoU_0.1',    avg_f1_score_10,   *kwargs)

            log_fn('mAP/validation_IoU_0.5',         avg_mAP_50,        *kwargs)
            log_fn('precision/validation_IoU_0.5',   avg_precision_50,  *kwargs)
            log_fn('recall/validation_IoU_0.5',      avg_recall_50,     *kwargs)
            log_fn('f1_score/validation_IoU_0.5',    avg_f1_score_50,   *kwargs)


        # Save the model using wandb if wandb is activated
        if self.use_wandb:
            dummy_input = torch.rand((1, self.input_channels, *self.input_size), device=self.device)
            model_filename = "model_final.onnx"
            log_dir = pjoin(wandb.config["logdir"], wandb.run.project, wandb.run.id)
            if not pexists(log_dir):
                os.makedirs(log_dir)
            self.to_onnx(pjoin(log_dir,model_filename), dummy_input, export_params=True)
            wandb.save(model_filename)

    def training_epoch_end(self, outputs):
        # Log the mAP every n epochs
        if self.current_epoch % (self.compute_metric_every_n_epochs * 2) == 0:
            avg_mAP_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["mAP"]]) for x in outputs]).mean()
            avg_precision_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["precision"]])
                                            for x in outputs]).mean()  # change this if n_classes > 2
            avg_recall_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["recall"]])
                                         for x in outputs]).mean()  # change this if n_classes > 2
            avg_f1_score_10 = torch.stack([torch.FloatTensor([x["log"]["metrics_10"]["f1_score"]]) for x in
                                           outputs]).mean()  # change this if n_classes > 2

            avg_mAP_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["mAP"]]) for x in outputs]).mean()
            avg_precision_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["precision"]]) for x in
                                            outputs]).mean()  # change this if n_classes > 2
            avg_recall_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["recall"]]) for x in
                                         outputs]).mean()  # change this if n_classes > 2
            avg_f1_score_50 = torch.stack([torch.FloatTensor([x["log"]["metrics_50"]["f1_score"]]) for x in
                                           outputs]).mean()  # change this if n_classes > 2

            # Logging
            log_fn = self.log if self.use_wandb else self.logger.experiment.add_scalar
            kwargs = [self.global_step] if not self.use_wandb else []
            log_fn('mAP/training_IoU_0.1',          avg_mAP_10,         *kwargs)
            log_fn('precision/training_IoU_0.1',    avg_precision_10,   *kwargs)
            log_fn('recall/training_IoU_0.1',       avg_recall_10,      *kwargs)
            log_fn('f1_score/training_IoU_0.1',     avg_f1_score_10,    *kwargs)

            log_fn('mAP/training_IoU_0.5',          avg_mAP_50,         *kwargs)
            log_fn('precision/training_IoU_0.5',    avg_precision_50,   *kwargs)
            log_fn('recall/training_IoU_0.5',       avg_recall_50,      *kwargs)
            log_fn('f1_score/training_IoU_0.5',     avg_f1_score_50,    *kwargs)

            l1_norm = self.compute_parameters_median_size()
            log_fn('hp_metric/parameter_sizes',     l1_norm,            *kwargs)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        predicted_locs, predicted_scores = self(batch["img"])

        filtered_results = self.detect_objects(predicted_locs, predicted_scores,
                                               min_score=self.min_score,
                                               max_overlap=self.max_overlap,
                                               top_k=self.top_k)

        det_boxes, det_label, det_scores = filtered_results

        return det_boxes, det_label, det_scores

    def configure_optimizers(self):
        biases = list()
        not_biases = list()
        for param_name, param in self.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        params = [{'params': biases, 'lr': 2 * self.lr}, {'params': not_biases}]
        # optimizer = torch.optim.SGD(params=params, lr=self.lr, momentum=0.9, weight_decay=0.0005)
        optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=0.0005)

        if self.scheduler != "none":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, verbose=False)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def compute_parameters_median_size(self):
        with torch.no_grad():
            l1_norm = sum(abs(p).sum() for p in self.parameters())
        return l1_norm

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                if v.requires_grad:
                    grads = v.grad
                    name = k
                    self.logger.experiment.add_histogram(tag="epoch/"+name, values=grads, global_step=self.trainer.global_step)


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcycz, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcycz = priors_cxcycz
        self.priors_xyz = cxcycz_to_xyz(priors_cxcycz)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.thresholding_mode = "soft" if type(self.threshold) == list else "hard"
        if self.thresholding_mode == "soft":
            assert (len(self.threshold) == 2)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the xxx prior boxes, a tensor of dimensions (N, xxx, 6)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, xxx, n_classes)
        :param boxes: true object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """

        def get_infos_by_chunking_data(n_objects, chunk_size, boxes):
            if n_objects == chunk_size:
                print()  # debug
            prior_for_each_object_by_split = list()
            overlap_for_each_prior_by_split = list()
            object_for_each_prior_by_split = list()

            for split in range((n_objects // chunk_size) + 1):
                if n_objects % chunk_size == 0 and split == n_objects // chunk_size:
                    break
                boxes_split = boxes[split * chunk_size:(split + 1) * chunk_size]

                overlap_split = find_jaccard_overlap3d(boxes_split, self.priors_xyz)  # (split_len, xxx)

                # For each prior in the split, find the object that has the maximum overlap
                overlap_for_each_prior_split, object_for_each_prior_split = overlap_split.max(dim=0)  # (xxx)

                # Back to the big picture to get the general index in the original tensor instead of the chunk index
                object_for_each_prior_split += (split * chunk_size)

                # We don't want a situation where an object is not represented in our positive (non-background) priors -
                # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
                # 2. All priors with the object may be assigned as background based on the threshold (0.5).

                # To remedy this -
                # First, find the prior that has the maximum overlap for each object.
                _, prior_for_each_object_split = overlap_split.max(dim=1)  # (split_len)

                prior_for_each_object_by_split.append(prior_for_each_object_split)
                overlap_for_each_prior_by_split.append(overlap_for_each_prior_split)
                object_for_each_prior_by_split.append(object_for_each_prior_split)

            # This is just concatenating the matched priors from all the chunks
            prior_for_each_object = torch.concat(prior_for_each_object_by_split)  # (n_objects)

            # This is a little more tricky. To find the best object for each prior as well as its overlap,
            # you must find the largest overlap from each chunk (overlap_for_each_prior_by_split (n_chunks, n_priors))
            # and its associated object index (object_for_each_prior (n_chunks, n_priors))
            # This ^ was done in the above for loop
            # Then you have to find the chunk (or overlap+index) that maximizes the overlap over all chunks
            # for every prior.
            overlap_for_each_prior = torch.concat(
                [o.view(1, -1) for o in overlap_for_each_prior_by_split])  # (n_chunks, n_priors)
            object_for_each_prior = torch.concat(
                [o.view(1, -1) for o in object_for_each_prior_by_split])  # (n_chunks, n_priors)

            # For every prior, find the max overlap and the index of its chunk
            overlap_for_each_prior, max_overlap_for_each_prior_chunk_indices = overlap_for_each_prior.max(
                dim=0)  # (n_priors)
            object_for_each_prior = object_for_each_prior.gather(0,
                                                                 max_overlap_for_each_prior_chunk_indices.view(1, -1))
            object_for_each_prior = object_for_each_prior.view(-1)  # (n_priors)

            return overlap_for_each_prior, object_for_each_prior, prior_for_each_object  # (n_priors), (n_priors), (n_objects)

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcycz.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 6), dtype=torch.float).to(device)  # (N, xxx, 6)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, xxx)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            if n_objects == 0:
                continue

            chunk_size = 100

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior, prior_for_each_object = get_infos_by_chunking_data(n_objects, chunk_size, boxes[i])

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)  # (xxx)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (xxx)

            ###########################################################

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            if self.thresholding_mode == "hard":
                label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (xxx)
            else:
                label_for_each_prior[overlap_for_each_prior < self.threshold[0]] = 0  # (xxx)
                label_for_each_prior[(overlap_for_each_prior >= self.threshold[0]) & (overlap_for_each_prior < self.threshold[1])] = -1

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcycz_to_gcxgcygcz(xyz_to_cxcycz(boxes[i][object_for_each_prior]), self.priors_cxcycz)  # (xxx, 6)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0  # (N, xxx)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & xxx)
        # So, if predicted_locs has the shape (N, xxx, 6), predicted_locs[positive_priors] will have (total positives, 6)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * xxx)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, xxx)
        conf_loss_all[true_classes < 0] = 0

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, xxx)
        conf_loss_neg[positive_priors] = 0.  # (N, xxx), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, xxx), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, xxx)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, xxx)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        # return conf_loss + self.alpha * loc_loss
        if torch.isnan(loc_loss):
            breakpoint()
            raise Exception("Loss is NaN")
        return conf_loss, loc_loss


if __name__ == '__main__':
    pass
    # model = LSSD3D(n_classes=2, input_channels=1).to(device)
    #
    # x = torch.rand(3, 1, 250, 300, 300).to(device)
    # locs, scores = model(x)
    #
    # gt_boxes = [torch.sort(torch.rand(1, 6), dim=1)[0].to(device) for i in range(batch_size)]
    # gt_labels = [torch.LongTensor([1]).to(device) for i in range(batch_size)]
    # criterion = MultiBoxLoss(model.priors_cxcycz)
    #
    # c, l = criterion(locs, scores, gt_boxes, gt_labels)
    # base = MobileNetBase(width_mult = 0.2, cube=True).cpu()
    # x = torch.rand(3, 1, 300, 300, 300).to(device)

    model = LSSD3D(n_classes=2, input_channels=1, input_size=(64,64,64)).to(device)
    # base = MobileNetBase(width_mult = 1., cube=True).to(device)
    x = torch.rand(3, 1, 64, 64, 64).to(device)
    gt_boxes = [torch.sort(torch.rand(1, 6), dim=1)[0].to(device) for i in range(3)]
    gt_labels = [torch.LongTensor([1]).to(device) for i in range(3)]
    # base(x)
    criterion = MultiBoxLoss(model.priors_cxcycz)
    criterion(*model(x), gt_boxes, gt_labels)


