import tensorflow as tf
import tensorflowvisu
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet


SOURCE_URL = 'https://storage.googleapis.com/cloud-deeplearning/kaggle_mnist_data/'
DOWNLOAD_DATASETS=True
DATA_DIR = '../input/'
KAGGLE_TRAIN_CSV = 'train.csv'
KAGGLE_TEST_CSV = 'test.csv'
SUBMISSION_FILE = 'submission_mnist_ann.csv'

# should sum up to 42000, the total number of images in train.csv
TRAIN_SIZE = 38000
VALID_SIZE = 2000
TEST_SIZE = 2000


def custom_kaggle_mnist():
    """
    downloads and parses mnist train dataset for kaggle digit recognizer
    parsing and one_hot copied https://www.kaggle.com/kakauandme/tensorflow-deep-nn
    """
    if DOWNLOAD_DATASETS:
        base.maybe_download(KAGGLE_TRAIN_CSV, DATA_DIR, SOURCE_URL + KAGGLE_TRAIN_CSV)

    # Import data from datasource, see https://www.kaggle.com/kakauandme/tensorflow-deep-nn
    # read training data from CSV file
    data = pd.read_csv(DATA_DIR + KAGGLE_TRAIN_CSV)

    from sklearn.utils import shuffle
    ## data = shuffle(data, random_state=42)

    images = data.iloc[:, 1:].values
    images = images.astype(np.float)
    images = np.reshape(images, (images.shape[0], 28, 28, 1))

    # convert from [0:255] => [0.0:1.0]
    ## images = np.multiply(images, 1.0 / 255.0)

    print('number of images in downloaded train dataset: {0[0]}'.format(images.shape))

    labels_flat = data.iloc[:, 0].values
    labels_count = np.unique(labels_flat).shape[0]

    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    ## labels = labels.astype(np.uint8)

    # split data into training & validation
    mnist_train_images = images[:TRAIN_SIZE]
    mnist_train_labels = labels[:TRAIN_SIZE]
    print('number of train images: {0[0]}'.format(mnist_train_images.shape))

    mnist_valid_images = images[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]
    mnist_valid_labels = labels[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]
    print('number of valid images: {0[0]}'.format(mnist_valid_images.shape))

    mnist_test_images = images[TRAIN_SIZE + VALID_SIZE:images.shape[0]]
    mnist_test_labels = labels[TRAIN_SIZE + VALID_SIZE:images.shape[0]]
    print('number of test images: {0[0]}'.format(mnist_test_images.shape))

    train = DataSet(mnist_train_images, mnist_train_labels, dtype=np.float32, reshape=False)
    valid = DataSet(mnist_valid_images, mnist_valid_labels, dtype=np.float32, reshape=False)
    test = DataSet(mnist_test_images, mnist_test_labels, dtype=np.float32, reshape=False)

    return base.Datasets(train=train, validation=valid, test=test)
