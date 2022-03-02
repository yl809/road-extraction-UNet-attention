import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from metrics1 import iou_coef, dice_coef
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from dataset1 import test_loader
from attention_unet import AttU_Net
from UNet import Unet
# from networks.dunet import Dunet
# from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

path = './saved_model/final.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(3,1)
model = model.to(device)
model.load_state_dict(torch.load(path))
model.eval()

IOU = 0.
DC = 0.
length = len(test_loader)
for index, (img, _, file_name) in enumerate(test_loader):
    img = img.to(device)

    img_GT = cv2.imread('./data_road/test/data/'+str(file_name[0]))
    label_GT = cv2.imread('./data_road/test/label/'+str(file_name[0]), cv2.IMREAD_GRAYSCALE)

    mask = model.forward(img)
    mask = mask.squeeze().cpu().data.numpy()

    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0

    DC += dice_coef(pred=mask, target=label_GT)
    IOU += iou_coef(pred=mask, target=label_GT)


    cv2.imwrite('./results/'+str(index)+'_pred.png', mask.astype(np.uint8))
    cv2.imwrite('./results/' + str(index) + '_img.png', img_GT)
    cv2.imwrite('./results/' + str(index) + '_label.png', label_GT)

print('mIOU: {}'.format(IOU/length))
print('mDC: {}'.format(DC/length))










