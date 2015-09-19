import os, struct
from array import array as std_array
import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig
from ConvLayer import ConvLayer
from FullyConnectedLayer import FullyConnectedLayer
from PoolingLayer import PoolingLayer
from SGD import encode_label
from SGD import SGD_train


def load_mnist_train(which, digits=np.arange(10)):
    """Load the mnist dataset and labels into a 3D array (height, width, num images)
    and column vector (1, num images) and return them both"""
    if which == "train":
        data_file = 'train-images-idx3-ubyte'
        label_file = 'train-labels-idx1-ubyte'
    elif which == "test":
        data_file = 't10k-images-idx3-ubyte'
        label_file = 't10k-labels-idx1-ubyte'

    images = open(data_file, 'rb')
    magic_num, size, rows, cols = struct.unpack('>IIII', images.read(16))
    images_arr = std_array("B", images.read())
    images.close()
    print "read images"

    labels = open(label_file, 'rb')
    magic_num, size = struct.unpack('>II', labels.read(8))
    labels_arr = std_array("b", labels.read())
    labels.close()
    print "read labels"

    ind = [k for k in xrange(size) if labels_arr[k] in digits]
    N = len(ind)

    dataset_imgs = np.zeros((rows, cols, N), dtype=np.uint8)
    dataset_labels = np.zeros((1,N), dtype=np.int8)
    for i in range(len(ind)):
        dataset_imgs[:,:,i] = np.array(images_arr[ind[i]*rows*cols:(ind[i]+1)*rows*cols]).reshape((rows, cols))
        dataset_labels[0,i] = labels_arr[ind[i]]

    return dataset_imgs, dataset_labels

def train():
    # load the training data
    images, labels = load_mnist_train("train")
    images = images.astype(np.float32)
    images /= 255.0

    grad_images = images[:,:,:] #training image subset
    grad_labels = labels.flatten().reshape((labels.size, 1))
    print grad_images[:,:,0:20].shape
    SGD_train(1, grad_images[:,:,0:30], grad_labels[0:30], 1.0e-1, 0.5, 20)

train()

def test_something():
    layer0 = ConvLayer((28, 28, 1), (9,9,2,2))
    layer0.W = 1
    print layer0.W

# test_something()
