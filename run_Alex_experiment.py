'''
train_X, train_y = extract_data('train_images.npy', 'train_labels.npy')
print("raw train dim")
print(train_X.shape)
print(train_y.shape)

print("\ntrain image 0 before process :\n", train_X[0].shape)
norm_train_X = np.zeros(shape=train_X.shape)
for i in range(0, train_X.shape[0]):
    img_tmp = train_X[i]
    result_tmp = normalize_image(img_tmp)
    norm_train_X[i] = result_tmp
print("train image 0 after process :\n", norm_train_X[0].shape)

n, chnl, d1, d2 = norm_train_X.shape
norm_FCN_train_X = norm_train_X.reshape((n, chnl*d1*d2))
print("fcn train dim")
print(norm_FCN_train_X.shape)

print('making loader for FCN train')
fcn_train_dataset = Dataset(norm_FCN_train_X, train_y)
fcn_train_loader = DataLoader(dataset=fcn_train_dataset,
                          batch_size=64,
                          shuffle=True)

print('making loader for CNN train')
cnn_train_dataset = Dataset(norm_train_X, train_y)
cnn_train_loader = DataLoader(dataset=cnn_train_dataset,
                          batch_size=64,
                          shuffle=True)



test_X, test_y = extract_data('test_images.npy', 'test_labels.npy')
print("\nraw test dim")
print(test_X.shape)
print(test_y.shape)

print("\ntest image 0 before process :\n", test_X[0].shape)
norm_test_X = np.zeros(shape=test_X.shape)
for i in range(0, test_X.shape[0]):
    img_tmp = test_X[i]
    result_tmp = normalize_image(img_tmp)
    norm_test_X[i] = result_tmp
print("test image 0 after process :\n", norm_test_X[0].shape)

n, chnl, d1, d2 = norm_test_X.shape
norm_FCN_test_X = norm_test_X.reshape((n, chnl*d1*d2))
print("fcn test dim")
print(norm_FCN_test_X.shape)

print('making loader for FCN test')
fcn_test_dataset = Dataset(norm_FCN_test_X, test_y)
fcn_test_loader = DataLoader(dataset=fcn_test_dataset,
                          batch_size=64,
                          shuffle=True)

print('making loader for CNN test')
cnn_test_dataset = Dataset(norm_test_X, test_y)
cnn_test_loader = DataLoader(dataset=cnn_test_dataset,
                          batch_size=64,
                          shuffle=True)



print('\ninit NN')
CNN_norm = ConvolutionalNN()

print('\ninit optimizer')
optimizer1 = optim.SGD(CNN_norm.parameters(), lr=0.001, momentum=0.9)

print('\ninit loss fn')
criterion = nn.CrossEntropyLoss()

print('\ntraining')
cnn_norm_test_accuracy, cnn_norm_train_accuracy, cnn_norm_loss = run_experiment(CNN_norm, cnn_train_loader, cnn_test_loader, criterion, optimizer1)

print('\nterm')
'''
