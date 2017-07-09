from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import tensorflow as tf
import kaggle_mnist as km

mnist = km.custom_kaggle_mnist()


# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
# *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
# Train (10, 100, 1000)
for index in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch_xs, Y_: batch_ys})

print(sess.run(accuracy, feed_dict={X: mnist.validation.images,
                                    Y_: mnist.validation.labels}))
# Test trained model before submission
print(sess.run(accuracy, feed_dict={X: mnist.test.images,
                                    Y_: mnist.test.labels}))

# kaggle test data
if km.DOWNLOAD_DATASETS:
    base.maybe_download(km.KAGGLE_TEST_CSV, km.DATA_DIR, km.SOURCE_URL + km.KAGGLE_TEST_CSV)
kaggle_test_images = pd.read_csv(km.DATA_DIR + km.KAGGLE_TEST_CSV).values.astype('float32')
kaggle_test_images = np.reshape(kaggle_test_images, (kaggle_test_images.shape[0], 28, 28, 1))

# convert from [0:255] => [0.0:1.0]
kaggle_test_images = np.multiply(kaggle_test_images, 1.0 / 255.0)

predictions_kaggle = sess.run(tf.argmax(tf.nn.softmax(Y), 1), feed_dict={X: kaggle_test_images})

with open(km.SUBMISSION_FILE, 'w') as submission:
    submission.write('ImageId,Label\n')
    for index, prediction in enumerate(predictions_kaggle):
        submission.write('{0},{1}\n'.format(index + 1, prediction))
    print("prediction submission written to {0}".format(km.SUBMISSION_FILE))
