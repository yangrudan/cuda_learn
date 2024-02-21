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


def show_landmarks(image, landmarks):
    """显示带有地标的图片"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001) # pause a bit so that plots are updated


plt.ion()
# interactive mode

landmarks_frame = pd.read_csv('./faces/face_landmarks.csv')
n = 65  # N是特征点的数量
img_name = landmarks_frame.iloc[n, 0]  # iloc：通过整数位置获得行和列的数据。
landmarks = landmarks_frame.iloc[n, 1:].to_numpy()
landmarks = landmarks.astype('float').reshape(-1, 2)  # It is quite simple, -1 means 'whatever it takes' to flatten. So, in the above example, a.reshape(2,-1) would mean 2*4, a.reshape(4,-1) would mean 4*2
print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

# Image name: person-7.jpg
# Landmarks shape: (68, 2)
# First 4 Landmarks: [[32. 65.]
#                     [33. 76.]
# [34. 86.]
# [34. 97.]]

plt.figure()
show_landmarks(io.imread(os.path.join('./faces/', img_name)),
               landmarks)
plt.show()
plt.savefig("person-7-mark.jpg")

