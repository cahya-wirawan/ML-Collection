from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import math
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet


SOURCE_URL = 'https://storage.googleapis.com/cloud-deeplearning/kaggle_mnist_data/'
DOWNLOAD_DATASETS=True
DATA_DIR = '../input/'
KAGGLE_TRAIN_CSV = 'train.csv'
KAGGLE_TEST_CSV = 'test.csv'
SUBMISSION_FILE = 'submission_mnist_cnn_all.csv'

# should sum up to 42000, the total number of images in train.csv
TRAIN_SIZE = 41000
VALID_SIZE = 500
TEST_SIZE = 500


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


mnist = custom_kaggle_mnist()

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
# *10 because  "mean" included an unwanted division by 10

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
## train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

# Train (10, 100, 1000)
BATCH_SIZE = 100
EPOCH_NUMBER = 30
RANGE_SIZE = int(EPOCH_NUMBER * TRAIN_SIZE/BATCH_SIZE)
for index in range(RANGE_SIZE):
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-index/decay_speed)

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    _accuracy, _train_step = sess.run([accuracy, train_step],
                                      feed_dict={X: batch_xs, Y_: batch_ys, lr: learning_rate})

    if index%(TRAIN_SIZE/BATCH_SIZE) == 0:
        print("Epoch {}, training accuracy: {}".format(int(1+index/(TRAIN_SIZE/BATCH_SIZE)), _accuracy))

print("Validation accuracy: {}".format(sess.run(accuracy, feed_dict={X: mnist.validation.images,
                                    Y_: mnist.validation.labels})))
# Test trained model before submission
print("Test accuracy: {}".format(sess.run(accuracy, feed_dict={X: mnist.test.images,
                                    Y_: mnist.test.labels})))

# kaggle test data
if DOWNLOAD_DATASETS:
    base.maybe_download(KAGGLE_TEST_CSV, DATA_DIR, SOURCE_URL + KAGGLE_TEST_CSV)
kaggle_test_images = pd.read_csv(DATA_DIR + KAGGLE_TEST_CSV).values.astype('float32')
kaggle_test_images = np.reshape(kaggle_test_images, (kaggle_test_images.shape[0], 28, 28, 1))

# convert from [0:255] => [0.0:1.0]
kaggle_test_images = np.multiply(kaggle_test_images, 1.0 / 255.0)

predictions_kaggle = sess.run(tf.argmax(tf.nn.softmax(Y), 1),
                              feed_dict={X: kaggle_test_images})

with open(SUBMISSION_FILE, 'w') as submission:
    submission.write('ImageId,Label\n')
    for index, prediction in enumerate(predictions_kaggle):
        submission.write('{0},{1}\n'.format(index + 1, prediction))
    print("prediction submission written to {0}".format(SUBMISSION_FILE))
