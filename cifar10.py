import pickle
import numpy as np
import os


def load(path_data_dir, method='color'):
    x_train, y_train = None, None
    for b in range(1, 6):
        with open(os.path.join(path_data_dir, 'data', 'cifar10', f"data_batch_{b}"), "rb") as f:
            batch = pickle.load(f, encoding='latin1')
            x_train = np.append(x_train, batch["data"].reshape((10000, 3072)), axis=0) if x_train is not None else batch["data"].reshape((10000, 3072))
            y_train = np.append(y_train, batch["labels"]) if y_train is not None else np.array(batch["labels"])

    with open(os.path.join(path_data_dir, 'data', 'cifar10', f"test_batch"), "rb") as f:
        batch = pickle.load(f, encoding='latin1')
        x_test = np.array(batch["data"].reshape((10000, 3072)))
        y_test = np.array(batch["labels"])

    if method == 'color':
        pass
    if method == 'average':
        x_train = x_train.reshape((50000, 3, 1024)).mean(axis=1).astype(np.uint8)
        x_test = x_test.reshape((10000, 3, 1024)).mean(axis=1).astype(np.uint8)
    if method == 'grayscale':
        factor = np.expand_dims(np.array([0.299, 0.587, 0.114]), 1)
        x_train = (x_train.reshape((50000, 3, 1024)) * factor).sum(axis=1).astype(np.uint8)
        x_test = (x_test.reshape((10000, 3, 1024)) * factor).sum(axis=1).astype(np.uint8)

    return x_train, y_train, x_test, y_test
