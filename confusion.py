import AlexNet as an
import cnn_xvalidation as cnnxv
import numpy as np
import warnings
import torch
import os
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import data_preproc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc

real_y = np.load('conf_real.npy')
print(len(real_y))
pred_y = np.load('conf_pred.npy')
print(len(pred_y))

confusion_mat = confusion_matrix(real_y, pred_y)
plt.figure()
cnnxv.plot_confusion_matrix(confusion_mat,
                          title='Confusion matrix, normalization')
plt.show()
