# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 00:41:18 2022

@author: Maxence Wynen
"""

import torch
from ssd3d import LSSD3D




for i in range(12):
    plt.imshow(out[0][i][144].cpu(), cmap="gray")
    plt.title(f"First Sequential Convolutional block (channel {i})")
    plt.colorbar()
    plt.show()

for i in range(25):
    plt.imshow(out[0][i][72].cpu(), cmap="gray")
    plt.title(f"First Block (channel {i})")
    plt.colorbar()
    plt.show()


for i in range(51):
    plt.imshow(out[0][i][36].cpu(), cmap="coolwarm")
    plt.title(f"Second Block (channel {i})")
    plt.colorbar()
    plt.show()