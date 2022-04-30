import os
import mnist_reader


def load(path_data_dir):
    path = os.path.join(path_data_dir, 'data/fashion')
    X_train, y_train = mnist_reader.load_mnist(path, kind='train')
    X_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
    return X_train, y_train, X_test, y_test
