"""
Copyright (c) Cookie Yang. All right reserved.
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
#用于更容易地进行csv解析
from skimage import io, transform
#用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# 忽略警告
import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)

plt.ion()
# interactive mode

landmarks_frame = pd.read_csv('./faces/face_landmarks.csv')
n = 65  # N是特征点的数量
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].to_numpy()
landmarks = landmarks.astype('float').reshape(-1, 2)
print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

# Image name: person-7.jpg
# Landmarks shape: (68, 2)
# First 4 Landmarks: [[32. 65.]
#                     [33. 76.]
# [34. 86.]
# [34. 97.]]
