from os.path import join as pjoin
from os.path import exists as exists
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pprint import pprint
import pandas as pd
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind

###### Plots per subject ######

# Boxplot mAP per subject
# Boxplot Prediction per subject
# Boxplot Recall per subject
# Boxplot F1 score per subject

metrics_per_subject_iou_50 = r"/home/wynen/PycharmProjects/MSLesions3D/data/predictions/#3k_64_n1-5_s6-14/1vb0zsnb/validation_set/min_score_0.0/aa_metrics_per_subject_(min_IoU=0.5).json"
with open(metrics_per_subject_iou_50, 'r') as file:
    metrics_per_subject_iou_50 = json.load(file)

df = pd.DataFrame(metrics_per_subject_iou_50).transpose()

# fig, axs = plt.subplots(ncols=4, figsize=(20,5))
# sns.boxplot(y='mAP', data=df, ax=axs[0])
# sns.boxplot(y='precision', data=df, ax=axs[1])
# sns.boxplot(y='recall', data=df, ax=axs[2])
# sns.boxplot(y='f1_score', data=df, ax=axs[3])
# plt.show()


###### General Plots ######
# Boxplot Volume of found boxes
# Boxplot Volume of unfound boxes

metrics_iou_10_sc_50 = r"/home/wynen/PycharmProjects/MSLesions3D/data/predictions/#3k_64_n1-5_s6-14/1vb0zsnb/validation_set/min_score_0.0/metrics_(min_IoU=0.1_min_score=0.5).json"
with open(metrics_iou_10_sc_50, 'r') as file:
    metrics_iou_10_sc_50 = json.load(file)

metrics_iou_50_sc_50 = r"/home/wynen/PycharmProjects/MSLesions3D/data/predictions/#3k_64_n1-5_s6-14/1vb0zsnb/validation_set/min_score_0.0/metrics_(min_IoU=0.5_min_score=0.5).json"
with open(metrics_iou_50_sc_50, 'r') as file:
    metrics_iou_50_sc_50 = json.load(file)

metrics_iou_50_sc_10 = r"/home/wynen/PycharmProjects/MSLesions3D/data/predictions/#3k_64_n1-5_s6-14/1vb0zsnb/validation_set/min_score_0.0/metrics_(min_IoU=0.5_min_score=0.1).json"
with open(metrics_iou_50_sc_10, 'r') as file:
    metrics_iou_50_sc_10 = json.load(file)

metrics_iou_10_sc_10 = r"/home/wynen/PycharmProjects/MSLesions3D/data/predictions/#3k_64_n1-5_s6-14/1vb0zsnb/validation_set/min_score_0.0/metrics_(min_IoU=0.1_min_score=0.1).json"
with open(metrics_iou_10_sc_10, 'r') as file:
    metrics_iou_10_sc_10 = json.load(file)


def plot_metric(metric):
    metrics = np.zeros((2, 2))
    metrics[0, 0] = metrics_iou_10_sc_10[metric]
    metrics[0, 1] = metrics_iou_10_sc_50[metric]
    metrics[1, 0] = metrics_iou_50_sc_10[metric]
    metrics[1, 1] = metrics_iou_50_sc_50[metric]
    ax = plt.axes()
    sns.heatmap(metrics, ax=ax, cmap="coolwarm")

    ax.set_title(f'{metric} per IoU threshold / minimum score threshold')
    plt.xlabel('IoU threshold', fontsize=10)
    plt.ylabel('Score threshold', fontsize=10)
    ax.set_xticklabels([0.1, 0.5])
    ax.set_yticklabels([0.1, 0.5])
    plt.show()


# plot_metric("mAP")
# plt.show()
plot_metric("precision")
plt.show()
# plot_metric("recall")
# plt.show()
# plot_metric("f1_score")
# plt.show()

################# VOLUMES ##################

# def v(lst):
#     return (np.array(lst) * (64 ** 3)) ** (1 / 3)
#
#
# p_10_10 = ttest_ind(v(metrics_iou_10_sc_10["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_10_sc_10["not_found_boxes_volumes_per_class"]), alternative='greater')
# p_10_50 = ttest_ind(v(metrics_iou_10_sc_50["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_10_sc_50["not_found_boxes_volumes_per_class"]), alternative='greater')
# p_50_10 = ttest_ind(v(metrics_iou_50_sc_10["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_50_sc_10["not_found_boxes_volumes_per_class"]), alternative='greater')
# p_50_50 = ttest_ind(v(metrics_iou_50_sc_50["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_50_sc_50["not_found_boxes_volumes_per_class"]), alternative='greater')
#
# fig, ax = plt.subplots(figsize=(15, 5))
# boxes = ax.boxplot([v(metrics_iou_10_sc_10["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_10_sc_10["not_found_boxes_volumes_per_class"]),
#                     v(metrics_iou_10_sc_50["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_10_sc_50["not_found_boxes_volumes_per_class"]),
#                     v(metrics_iou_50_sc_10["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_50_sc_10["not_found_boxes_volumes_per_class"]),
#                     v(metrics_iou_50_sc_50["found_boxes_volumes_per_class"]),
#                     v(metrics_iou_50_sc_50["not_found_boxes_volumes_per_class"]), ],
#                    positions=[1, 1.6, 2.5, 3.1, 4, 4.6, 5.5, 6.1],
#                    labels=['Found', 'Not Found', 'Found', 'Not Found', 'Found', 'Not Found', 'Found', 'Not Found'],
#                    patch_artist=True
#                    )
# for i, box in enumerate(boxes["boxes"]):
#     if i % 2 == 0:
#         box.set(color='limegreen')
#     else:
#         box.set(color='tomato')
#
# for i, box in enumerate(boxes["medians"]):
#     box.set(color='black')
#
# ax.set_xticks([1.3, 2.8, 4.3, 5.8])
# ax.set_xticklabels(["IoU > 0.1\nScore > 0.1\np-value(v(F) > v(NF))={:.3g}".format(p_10_10.pvalue),
#                     "IoU > 0.1\nScore > 0.5\np-value(v(F) > v(NF))={:.3g}".format(p_10_50.pvalue),
#                     "IoU > 0.5\nScore > 0.1\np-value(v(F) > v(NF))={:.3g}".format(p_50_10.pvalue),
#                     "IoU > 0.5\nScore > 0.5\np-value(v(F) > v(NF))={:.3g}".format(p_50_50.pvalue)])
# green_patch = mpatches.Patch(color='limegreen', label='Found boxes')
# red_patch = mpatches.Patch(color='tomato', label='Not found boxes')
# ax.legend(handles=[green_patch, red_patch], loc='upper right')
# ax.set_title("Boxes Volume")
# ax.set_ylabel("Edge length in voxels")