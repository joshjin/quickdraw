import AlexNet as an
import cnn_xvalidation as cnnxv
import numpy as np
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms

# an.load_data_for_alex('X.npy', 'y.npy')
'''
X_train, y_train, X_test, y_test = cnnxv.extract_data(os.path.join("large_small_data", "data"))
np.save('X_train_25_weight.npy', X_train)
np.save('y_train_25_weight.npy', y_train)
np.save('X_test_25_weight.npy', X_test)
np.save('y_test_25_weight.npy', y_test)
'''



train_X, train_25_y = cnnxv.load_data('X_train_25_weight.npy', 'y_train_25_weight.npy')
print("raw train dim")
print(train_X.shape)
print(train_25_y.shape)

tmp = np.copy(train_25_y)
train_5_y = np.floor(np.divide(tmp, 5))

cnn_train_dataset = cnnxv.Dataset(train_X, train_25_y)
cnn_train_loader = DataLoader(dataset=cnn_train_dataset,
                          batch_size=64,
                          shuffle=True)

test_X, test_25_y = cnnxv.load_data('X_test_25_weight.npy', 'y_test_25_weight.npy')
print("raw test dim")
print(test_X.shape)
print(test_25_y.shape)


tmp = np.copy(test_25_y)
test_5_y = np.floor(np.divide(tmp, 5))

cnn_test_dataset = cnnxv.Dataset(test_X, test_25_y)
cnn_test_loader = DataLoader(dataset=cnn_test_dataset,
                          batch_size=64,
                          shuffle=True)

# print('largest class: ', str(np.amax(train_5_y)))
# print('largest class: ', str(np.amax(test_5_y)))


'''
train_v = []
test_v = []
print('For train')
for i in range(train_5_y.shape[0]):
    if i%100 == 0:
        train_v.append(train_5_y[i])
print(train_v)
print('For test')
for j in range(test_5_y.shape[0]):
    if j%100 == 0:
        test_v.append(test_5_y[j])
print(test_v)

print(test_25_y[0:10])
print(train_25_y[0:10])
'''

print('\ninit NN')
CNN = cnnxv.LeNet5()

print('\ninit optimizer')
optimizer1 = optim.SGD(CNN.parameters(), lr=0.005, momentum=0.9)

print('\ninit loss fn')
criterion = nn.CrossEntropyLoss()

print('\ntraining')
cnn_norm_test_accuracy, cnn_norm_train_accuracy, cnn_norm_loss = cnnxv.run_experiment_with_subclass(CNN, cnn_train_loader, cnn_test_loader, criterion, optimizer1)

print('\nterm')

