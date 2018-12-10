import AlexNet as an
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
X_train, y_train, X_test, y_test = an.extract_data(os.path.join("519_refined_data", "data"))
np.save('X_train_small.npy', X_train)
np.save('y_train_small.npy', y_train)
np.save('X_test_small.npy', X_test)
np.save('y_test_small.npy', y_test)



train_X, train_y = an.load_data_for_alex('X_train_small.npy', 'y_train_small.npy')
print("raw train dim")
print(train_X.shape)
print(train_y.shape)

alex_train_dataset = an.Dataset(train_X, train_y)
alex_train_loader = DataLoader(dataset=alex_train_dataset,
                          batch_size=64,
                          shuffle=True)

test_X, test_y = an.load_data_for_alex('X_test_small.npy', 'y_test_small.npy')
print("raw train dim")
print(test_X.shape)
print(test_y.shape)

alex_test_dataset = an.Dataset(test_X, test_y)
alex_test_loader = DataLoader(dataset=alex_test_dataset,
                          batch_size=64,
                          shuffle=True)

print('\ninit NN')
AlexNN = an.AlexNN()

print('\ninit optimizer')
optimizer1 = optim.SGD(AlexNN.parameters(), lr=0.005, momentum=0.9)

print('\ninit loss fn')
criterion = nn.CrossEntropyLoss()

print('\ntraining')
cnn_norm_test_accuracy, cnn_norm_train_accuracy, cnn_norm_loss = an.run_experiment(AlexNN, alex_train_loader, alex_test_loader, criterion, optimizer1)

print('\nterm')

