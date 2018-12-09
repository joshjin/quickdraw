import AlexNet as an
import numpy as np
import os

# an.load_data_for_alex('X.npy', 'y.npy')

X_train, y_train, X_test, y_test = an.extract_data(os.path.join("519_refined_data", "data"))
np.save('X_train_small.npy', X_train)
np.save('y_train_small.npy', y_train)
np.save('X_test_small.npy', X_test)
np.save('y_test_small.npy', y_test)