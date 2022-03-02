import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader,Dataset
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio


root = './data_road/'
class RoadDataset(Dataset):

    def __init__(self, phase='train'):
        self.phase = phase

    def __len__(self):
        return len(os.listdir(root+self.phase+'/'+'data'))


    def __getitem__(self, index):
        file_name = os.listdir(os.path.join(root+self.phase, 'data'))[index]

        img = cv2.imread(os.path.join(root+self.phase, 'data', file_name))
        mask = cv2.imread(os.path.join(root+self.phase, 'label', file_name), cv2.IMREAD_GRAYSCALE)

        mask = np.expand_dims(mask, axis=2)

        # 归一化
        # 讲图像从[0,255]规范化到[-1.6,1.6]
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img, mask, file_name

train_dataset = RoadDataset(phase='train')
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)
valid_dataset = RoadDataset(phase='valid')
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=True
)
test_dataset = RoadDataset(phase='test')
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
)