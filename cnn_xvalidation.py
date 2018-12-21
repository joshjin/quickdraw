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

warnings.filterwarnings("ignore")


############################################################
# Run Experiment
############################################################

large_class = {'cat', 'stove', 'computer', 'flower', 'car'}


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
    large_train_len = 3000
    large_test_len = 200
    short_train_len = 200
    short_test_len = 200
    for file in os.listdir(path):
        print('class: ', str(file))
        print(os.path.join(path, file))
        flag = ''
        if str(file).split(sep='.')[0] in large_class:
            df = data_preproc.read_data(os.path.join(path, file), 10000)
            train_len = large_train_len
            test_len = large_test_len
            flag = 'large'
        else:
            df = data_preproc.read_data(os.path.join(path, file), 2000)
            train_len = short_train_len
            test_len = short_test_len
            flag = 'small'
        df = data_preproc.process_df(df)
        df = data_preproc.convert_df_into_image(df).reset_index()
        data_len =df.shape[0]
        print(df.shape)
        if flag == 'large':
            X_train_np_tmp = np.zeros(shape=(min(data_len, train_len), 42, 42))
            y_train_np_tmp = np.zeros(shape=(min(data_len, train_len),))
            X_test_np_tmp = np.zeros(shape=(min(data_len, test_len), 42, 42))
            y_test_np_tmp = np.zeros(shape=(min(data_len, test_len),))
        else:
            X_train_np_tmp = np.zeros(shape=(train_len*15, 42, 42))
            y_train_np_tmp = np.zeros(shape=(train_len*15))
            X_test_np_tmp = np.zeros(shape=(min(data_len, test_len), 42, 42))
            y_test_np_tmp = np.zeros(shape=(min(data_len, test_len),))
        for i in range(0, min(data_len, (train_len+test_len))):
            if flag == 'large':
                if i < train_len:
                    # if i % 100 == 0:
                    #     print(str(i), ' iterations')
                    X_tmp = df.loc[i, 'image']
                    y_tmp = df.loc[i, 'word']
                    X_tmp_square = X_tmp.reshape(42, 42)
                    X_train_np_tmp[i,:,:]=X_tmp_square
                    y_num = data_preproc.cls_ordered_small[y_tmp]
                    y_train_np_tmp[i] = y_num
                else:
                    X_tmp = df.loc[i, 'image']
                    y_tmp = df.loc[i, 'word']
                    X_tmp_square = X_tmp.reshape(42, 42)
                    j = i - train_len
                    X_test_np_tmp[j, :, :] = X_tmp_square
                    y_num = data_preproc.cls_ordered_small[y_tmp]
                    y_test_np_tmp[j] = y_num
            else:
                if i < train_len:
                    # if i % 100 == 0:
                    #     print(str(i), ' iterations')
                    X_tmp = df.loc[i, 'image']
                    y_tmp = df.loc[i, 'word']
                    X_tmp_square = X_tmp.reshape(42, 42)
                    y_num = data_preproc.cls_ordered_small[y_tmp]
                    for j in range(0,15):
                        X_train_np_tmp[i+j*200,:,:]=X_tmp_square
                        y_train_np_tmp[i+j*200] = y_num
                else:
                    X_tmp = df.loc[i, 'image']
                    y_tmp = df.loc[i, 'word']
                    X_tmp_square = X_tmp.reshape(42, 42)
                    j = i - train_len
                    X_test_np_tmp[j, :, :] = X_tmp_square
                    y_num = data_preproc.cls_ordered_small[y_tmp]
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
    print(X.shape)
    for i in range(X.shape[0]):
        X_[i, 0, :, :] = X[i,:,:]
    print(X_.shape)
    print(y.shape)
    return X_, y


def load_data_for_alex(X_file, y_file):
    X = np.load(X_file)
    y = np.load(y_file)
    print("successfully loaded data")
    X_ = np.zeros((X.shape[0], 1, 227, 227))
    for i in range(X.shape[0]):
        if i%3000==0:
            print(str(i/300), " percent done")
        img_tmp = cv2.resize(np.reshape(X[i,:,:], (42, 42)), dsize=(42*5, 42*5), interpolation=cv2.INTER_CUBIC)
        X_[i, 0, 10:220, 10:220] = img_tmp
    print(X_.shape)
    print(y.shape)
    # img = np.reshape(X_[2,0,:,:], (227, 227))
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return X_, y


# X, y = extract_data(os.path.join("519_refined_data", "data"))
# np.save('X.npy', X)
# np.save('y.npy', y)


class LeNet5(nn.Module):
    """
        (1) Use self.conv1 as the variable name for your first convolutional layer
        (2) Use self.pool as the variable name for your pooling layer
        (3) User self.conv2 as the variable name for your second convolutional layer
        (4) Use self.fc1 as the variable name for your first fully connected layer
        (5) Use self.fc2 as the variable name for your second fully connected layer
        (6) Use self.fc3 as the variable name for your third fully connected layer
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 7, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 16, 400)
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 25)

    def forward(self, x):
        z1 = F.relu(self.conv1(x))
        z2 = self.pool(z1)
        z3 = F.relu(self.conv2(z2))
        z4 = self.pool(z3)
        z5 = z4.view(z4.size(0), -1)
        z6 = F.relu(self.fc1(z5))
        z7 = F.relu(self.fc2(z6))
        z8 = F.sigmoid(self.fc3(z7))
        return z8


class AlexNN(nn.Module):
    """
        (1) Use self.conv1 as the variable name for your first convolutional layer
        (2) Use self.pool as the variable name for your pooling layer
        (3) User self.conv2 as the variable name for your second convolutional layer
        (4) Use self.fc1 as the variable name for your first fully connected layer
        (5) Use self.fc2 as the variable name for your second fully connected layer
        (6) Use self.fc3 as the variable name for your third fully connected layer
    """

    def __init__(self):
        super(AlexNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        z1 = F.relu(self.conv1(x))
        z2 = self.pool1(z1)
        z3 = F.relu(self.conv2(z2))
        z4 = self.pool2(z3)
        z5 = F.relu(self.conv3(z4))
        z6 = F.relu(self.conv4(z5))
        z7 = self.pool3(z6)
        z8 = z7.view(z7.size(0), -1)
        z9 = F.relu(self.fc1(z8))
        z10 = F.relu(self.fc2(z9))
        z11 = F.sigmoid(self.fc3(z10))
        return z11


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

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




def run_experiment_with_subclass(neural_network, train_loader, test_loader, loss_function, optimizer):
    real_label = []
    pred_label = []
    max_epochs = 13
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
            if epoch == max_epochs-1:
                real_label = real_label + test_labels_np.tolist()
                pred_label = pred_label + test_predicted_np.tolist()
        test_accuracy[epoch] = test_acc_tmp / test_count
        print("epoch: ", str(epoch + 1), "test_acc: ", test_accuracy[epoch])

    print(len(real_label))
    np.save('conf_real.npy', real_label)
    np.save('conf_pred.npy', pred_label)

    # confusion_mat = confusion_matrix(test_labels, test_predicted)
    # plt.figure()
    # plot_confusion_matrix(confusion_mat,
    #                       title='Confusion matrix, without normalization')

    # print("final training accuracy: ", test_accuracy[max_epochs-1])
    return test_accuracy, train_accuracy, loss_np

