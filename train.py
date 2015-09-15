import os, struct
import numpy as np
from array import array as std_array
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig
from ConvLayer import ConvLayer
from FullyConnectedLayer import FullyConnectedLayer
from PoolingLayer import PoolingLayer

def load_mnist(digits=np.arange(10)):
    """Load the mnist dataset and labels into a 3D array (height, width, num images)
    and column vector (1, num images) and return them both"""

    images = open('train-images-idx3-ubyte', 'rb')
    magic_num, size, rows, cols = struct.unpack('>IIII', images.read(16))
    images_arr = std_array("B", images.read())
    images.close()
    print "read images"

    labels = open('train-labels-idx1-ubyte', 'rb')
    magic_num, size = struct.unpack('>II', labels.read(8))
    labels_arr = std_array("b", labels.read())
    labels.close()
    print "read labels"

    ind = [k for k in xrange(size) if labels_arr[k] in digits]
    N = len(ind)

    train_dataset_imgs = np.zeros((rows, cols, N), dtype=np.uint8)
    train_dataset_labels = np.zeros((1,N), dtype=np.int8)
    for i in range(len(ind)):
        train_dataset_imgs[:,:,i] = np.array(images_arr[ind[i]*rows*cols:(ind[i]+1)*rows*cols]).reshape((rows, cols))
        train_dataset_labels[0,i] = labels_arr[ind[i]]

    return train_dataset_imgs, train_dataset_labels

def J(output, label):
    """Our error function--in this case cross-entropy"""
    sum_err = 0
    for i in range(0, 10):
        if label == i:
            sum_err -= np.log(output[i])
    return sum_err

def testGradient():
    """Test the backprop implementation by checking the gradients on a small network"""

    # load the training data
    images, labels = load_mnist()
    images /= 255.0

    grad_images = images[:,:,0:10] #use 10 image subset for gradient checking
    grad_labels = labels[0,0:10] #respective labels for the images--going to have to encode these labels

    # create a small network, 1 conv layer + 1 pooling layer + 1 fully connected softmax

    # convolutional layer, taking in a 28x28 image, using 2 9x9 filters
    # output should be 2 28-9+1x28-9+1 = 2 20x20 feature maps in a (20, 20, 2) form
    layer0 = ConvLayer(grad_images[:,:,0].reshape((28,28,1)), (28, 28, 1), (9, 9, 2, 1))
    print "initalized convolutional layer"
    layer0.forwardprop(grad_images[:,:,0].reshape((28,28,1)))
    print "finished forward pass of convolutional layer"

    # pooling layer, taking in 2 20x20 feature maps
    # output should be 2 10x10 feature maps (though may want to downsample 5x for gradient check)
    layer1 = PoolingLayer(layer0.output, (20, 20, 2))
    print "initialized pooling layer"
    layer1.downsample(layer0.output, (20, 20, 2))
    print "finished forward pass of pooling layer"

    # fully-connected softmax layer, taking in 2 10x10 feature maps (if downsampled by 2)
    # or taking in 2 4x4 feature maps (if downsampled by 5)
    # either way, flattened into a long input vector
    full_conn_input = layer1.output.flatten()
    layer2 = FullyConnectedLayer(full_conn_input.reshape((full_conn_input.size, 1)), full_conn_input.size, 10)
    print "initialized fully-conn layer"
    layer2.softmax_output(full_conn_input.reshape((full_conn_input.size, 1)))
    print "finished forward pass of fully-conn layer"

    # perform backpropagation
    target = np.zeros((10,1))
    for i in range(0, 10):
        if grad_labels[i] == 1:
            target[i] = 1
    layer2.backprop(0, 0, target)
    print "finished layer 2 backprop"
    layer1.upsample(layer2, 0)
    print "finished layer 1 backprop"
    layer0.backprop(layer1)
    print "finished layer 0 backprop"

    # # after initialization, finish training
    # for i in range(1, grad_labels.size):
    #     # forward propagation
    #     layer0.forwardprop(grad_images[:,:,i].reshape((28,28,1)))
    #     layer1.downsample(layer0.output, (20,20,2))
    #     full_conn_input = layer1.output.flatten()
    #     layer2.softmax_output(full_conn_input.reshape((full_conn_input.size, 1)))
    #
    #     # backpropagation
    #     target = np.zeros((10,1))
    #     for j in range(0,10):
    #         if grad_labels[i] == 1:
    #             target[i] = 1
    #     layer2.backprop(0, 0, target)
    #     layer1.upsample(layer2, 0)
    #     layer0.backprop(layer1)

    # check the gradient
    epsilon = 1.0e-4
    layer0_check = layer0
    layer1_check = layer1
    layer2_check = layer2

    layer0_w_vec = layer0.W.flatten()
    layer0_bias_vec = layer0.bias.flatten()
    layer0_gradw = layer0.gradient_w.flatten()
    layer0_gradb = layer0.gradient_b.flatten()

    layer2_w_vec = layer2.W.flatten()
    layer2_bias_vec = layer2.bias.flatten()
    layer2_gradw = layer2.gradient_w.flatten()
    layer2_gradb = layer2.gradient_b.flatten()

    w_vec = np.concatenate((layer0_w_vec, layer0_bias_vec, layer2_w_vec, layer2_bias_vec))
    backprop_vec = np.concatenate((layer0_gradw, layer0_gradb, layer2_gradw, layer2_gradb))
    print layer0_gradw
    gradient_check = np.zeros(w_vec.size)
    for i in range(0, w_vec.size):
        pos = w_vec
        pos[i] += epsilon
        neg = w_vec
        neg[i] -= epsilon
        # feed-forward to get J(w+e), J(w-e), subtract and calculate gradient
        # J(w+e)
        layer0_check.W = pos[0:layer0_w_vec.size].reshape(layer0.filter_shape)
        layer0_check.bias = pos[layer0_w_vec.size : layer0_w_vec.size+layer0_bias_vec.size].reshape(layer0.bias_shape)

        layer2_check.W = pos[layer0_w_vec.size+layer0_bias_vec.size : layer0.W.size+layer0.bias.size+layer2_w_vec.size].reshape(layer2.W.shape)
        layer2_check.bias = pos[layer0.W.size+layer0.bias.size+layer2_w_vec.size:].reshape(layer2.bias.shape)

        layer0_check.forwardprop(grad_images[:,:,0].reshape((28,28,1)))
        layer1_check.downsample(layer0_check.output, (20,20,2))
        full_conn_input = layer1.output.flatten()
        layer2_check.softmax_output(full_conn_input.reshape((full_conn_input.size, 1)))

        pos_out = J(layer2_check.output, grad_labels[0])
        # J(w-e)
        layer0_check.W = neg[0:layer0_w_vec.size].reshape(layer0.filter_shape)
        layer0_check.bias = neg[layer0_w_vec.size : layer0_w_vec.size+layer0_bias_vec.size].reshape(layer0.bias_shape)

        layer2_check.W = neg[layer0_w_vec.size+layer0_bias_vec.size : layer0.W.size+layer0.bias.size+layer2_w_vec.size].reshape(layer2.W.shape)
        layer2_check.bias = neg[layer0.W.size+layer0.bias.size+layer2_w_vec.size:].reshape(layer2.bias.shape)

        layer0_check.forwardprop(grad_images[:,:,0].reshape((28,28,1)))
        layer1_check.downsample(layer0_check.output, (20,20,2))
        full_conn_input = layer1.output.flatten()
        layer2_check.softmax_output(full_conn_input.reshape((full_conn_input.size, 1)))

        neg_out = J(layer2_check.output, grad_labels[0])
        # compute gradient for i
        gradient_check[i] = (pos_out - neg_out)/(2*epsilon)

    # print gradient_check
    print gradient_check[0:layer0_w_vec.size]

testGradient()
