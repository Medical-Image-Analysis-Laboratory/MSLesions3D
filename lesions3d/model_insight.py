# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:21:07 2022

@author: Maxence Wynen
"""

import torch
import pytorch_lightning as pl
from ssd3d import *
# import seaborn as sns
import  matplotlib.pyplot as plt



model_path = r"C:\Users\Cristina\Desktop\MSLesions3D\tensorboard\lesions\zebardi\no_augmentation\checkpoints\epoch=249-step=3250.ckpt"

# model = LSSD3D.load_from_checkpoint(checkpoint_path=model_path)
model = LSSD3D(n_classes=1 + 1, input_channels=1, lr=5e-4, width_mult=0.4, scheduler="none", batch_size=8,
                   comments="comments", compute_metric_every_n_epochs=5)
model.init()

params_per_layer = {}
zeros_per_layer = {}

for n,p in model.named_parameters():

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
            plt.title(f'{part} layer {layer} // # zeros = {len([abs(x) for x in data if x < 1e-15])} out of {len(data)}')
            plt.show()



