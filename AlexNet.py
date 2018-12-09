import pandas as pd
import numpy as np
from ast import literal_eval
import warnings
import torch
import os
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.transform import resize, rescale
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import data_preproc
from torchvision import transforms

warnings.filterwarnings("ignore")


############################################################
# Run Experiment
############################################################

def print_img(im, label, size):
    data = im.copy().astype(int)
    print(data.shape)
    print(data)
    i_data = np.zeros((size, size, 3), dtype=np.uint8)
    for j in range(0,3):
        i_data[:,:,j] = data*200
    img = Image.fromarray(i_data, 'RGB')
    img.save(str(label) + ".png")

def extract_data(path):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for file in os.listdir(path):
        print('class: ', str(file))
        print(os.path.join(path, file))
        df = data_preproc.read_data(os.path.join(path, file), 10000)
        df = data_preproc.process_df(df)
        df = data_preproc.convert_df_into_image(df).reset_index()
        data_len =df.shape[0]
        print(df.shape)
        X_train_np_tmp = np.zeros(shape=(min(data_len, 3000), 42, 42))
        y_train_np_tmp = np.zeros(shape=(min(data_len, 3000),))
        X_test_np_tmp = np.zeros(shape=(min(data_len, 200), 42, 42))
        y_test_np_tmp = np.zeros(shape=(min(data_len, 200),))
        for i in range(0, min(data_len, 3200)):
            if i < 3000:
                if i % 500 == 0:
                    print(str(i), ' iterations')
                X_tmp = df.loc[i, 'image']
                y_tmp = df.loc[i, 'word']
                X_tmp_square = X_tmp.reshape(42, 42)
                X_train_np_tmp[i,:,:]=X_tmp_square
                y_num = data_preproc.cls_dict[y_tmp]
                y_train_np_tmp[i] = y_num
            else:
                X_tmp = df.loc[i, 'image']
                y_tmp = df.loc[i, 'word']
                X_tmp_square = X_tmp.reshape(42, 42)
                j = i - 3000
                X_test_np_tmp[j, :, :] = X_tmp_square
                y_num = data_preproc.cls_dict[y_tmp]
                y_test_np_tmp[j] = y_num
        X_train = X_train + X_train_np_tmp.tolist()
        y_train = y_train + y_train_np_tmp.tolist()
        X_test = X_test + X_test_np_tmp.tolist()
        y_test = y_test + y_test_np_tmp.tolist()
        print(len(y_train))
        print(len(y_test))
    return X_train, y_train, X_test, y_test


def load_data(X_file, y_file):
    X = np.load(X_file)
    y = np.load(y_file)
    X_ = np.zeros((X.shape[0], 1, X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        X_[i, 1, :, :] =X[i,:,:]
    print(X_.shape)
    print(y.shape)
    return X, y


def load_data_for_alex(X_file, y_file):
    X = np.load(X_file)
    X_len = X.shape[0]
    # X_large = np.zeros((X_len, 1, 5*42, 5*42))
    # for i in range(X_len):
    #     X_large[i, 1, :, :] =

    # for i in range(X_len):
    #     X_large[i,1,:,:] = resize(X[i,:,:], (42*5, 42*5))
    # y = np.load(y_file)
    #small_img = np.reshape(X[1,:,:], (42, 42))
    #print_img(small_img, "small_test", 42)
    #small_img = cv2.imread('small_test.png')
    img = cv2.resize(np.reshape(X[1,:,:], (42, 42)), dsize=(42*5, 42*5), interpolation=cv2.INTER_CUBIC)
    # img = resize(np.reshape(X[1,:,:]*200, (42, 42)), (5*42, 5*42), PIL.Image.ANTIALIAS)
    # print_img(img, "large_test", 42*5)
    # return X, y
    cv2.imshow('image', img)
    cv2.waitKey(0)


# X, y = extract_data(os.path.join("519_refined_data", "data"))
# np.save('X.npy', X)
# np.save('y.npy', y)


class ConvolutionalNN(nn.Module):
    """
        (1) Use self.conv1 as the variable name for your first convolutional layer
        (2) Use self.pool as the variable name for your pooling layer
        (3) User self.conv2 as the variable name for your second convolutional layer
        (4) Use self.fc1 as the variable name for your first fully connected layer
        (5) Use self.fc2 as the variable name for your second fully connected layer
        (6) Use self.fc3 as the variable name for your third fully connected layer
    """

    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, 3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 16, 3, stride=1, padding=0)
        self.fc1 = nn.Linear(13 * 13 * 16, 130)
        self.fc2 = nn.Linear(130, 72)
        self.fc3 = nn.Linear(72, 10)

    def forward(self, x):
        z1 = F.relu(self.conv1(x))
        z2 = self.pool(z1)
        z3 = F.relu(self.conv2(z2))
        z4 = z3.view(z3.size(0), -1)
        z5 = F.relu(self.fc1(z4))
        z6 = F.relu(self.fc2(z5))
        z7 = F.sigmoid(self.fc3(z6))

        return z7


class Dataset(Dataset):
    """CIFAR-10 image dataset."""

    def __init__(self, X, y, transformations=None):
        self.len = len(X)
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def run_experiment(neural_network, train_loader, test_loader, loss_function, optimizer):
    max_epochs = 100
    loss_np = np.zeros((max_epochs))
    train_accuracy = np.zeros((max_epochs))
    test_accuracy = np.zeros((max_epochs))

    for epoch in range(max_epochs):
        train_count = 0
        train_acc_tmp = 0.0
        los_tmp = 0.0
        for i, data in enumerate(train_loader, 0):
            train_count += 1
            train_inputs, train_labels = data
            train_inputs, train_labels = Variable(train_inputs), Variable(train_labels)
            train_y_pred = neural_network(train_inputs)
            loss = loss_function(train_y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct = 0
            train_total = train_labels.size(0)
            _, train_predicted = torch.max(train_y_pred.data, 1)
            train_predicted_np = train_predicted.numpy()
            train_labels_np = train_labels.data.numpy()
            train_correct += (train_predicted_np == train_labels_np).sum().item()
            train_acc_tmp += float(train_correct) / float(train_total)
            los_tmp += loss.data[0]
        train_accuracy[epoch] = train_acc_tmp / train_count
        loss_np[epoch] = los_tmp / train_count
        print("epoch: ", str(epoch + 1), "train_loss: ", loss_np[epoch], "train_acc: ", train_accuracy[epoch])
        test_count = 0
        test_acc_tmp = 0.0
        for i, data in enumerate(test_loader, 0):
            test_count += 1
            test_inputs, test_labels = data
            test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)
            test_correct = 0
            test_total = test_labels.size(0)
            test_y_pred = neural_network(test_inputs)
            _, test_predicted = torch.max(test_y_pred.data, 1)
            test_predicted_np = test_predicted.numpy()
            test_labels_np = test_labels.data.numpy()
            test_correct += (test_predicted_np == test_labels_np).sum().item()
            test_acc_tmp += float(test_correct) / float(test_total)
        test_accuracy[epoch] = test_acc_tmp / test_count
        print("epoch: ", str(epoch + 1), "test_acc: ", test_accuracy[epoch])

    # print("final training accuracy: ", test_accuracy[max_epochs-1])
    return test_accuracy, train_accuracy, loss_np


