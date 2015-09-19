import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig
from ConvLayer import ConvLayer
from FullyConnectedLayer import FullyConnectedLayer
from PoolingLayer import PoolingLayer

def encode_label(label):
    encoded_label = np.zeros((10,1))
    for i in range(0, 10):
        if i == label:
            encoded_label[i] = 1
    return encoded_label

def J(output, label):
    """Our error function--in this case cross-entropy"""
    sum_err = 0
    for i in range(0, 10):
        if label == i:
            sum_err -= np.log(output[i])
    return sum_err

def SGD_train(minibatch_size, data, labels, alpha, momentum, epochs):
    """Train the network with stochastic gradient descent

    :type minibatch_size: an integer
    :param minibatch_size: the size of the minibatches (usually something like 256)

    :type data: 3D matrix height x width x num training data pts.
    :param data: A 3D matrix that contains all of the training data points of the set

    :type labels: num training data pts x 1 vector
    :param labels: the labels for each image

    :type alpha: float
    :param alpha: the learning rate

    :type momentum: float
    :param momentum: the momentum

    :type epochs: an integer
    :param epochs: the number of epochs (ie. iterations) through the training
    """

    it = 0
    # convolutional layer, taking in a 28x28 image, using 2 9x9 filters
    # output should be 2 28-9+1x28-9+1 = 2 20x20 feature maps in a (20, 20, 2) form
    layer0 = ConvLayer((28, 28, 1), (9,9,2))
    print "initialized convolutional layer"
    # pooling layer, taking in 2 20x20 feature maps
    # output should be 2 10x10 feature maps
    layer1 = PoolingLayer((20, 20, 2))
    print "initialized pooling layer"
    # fully-connected softmax layer, taking in 2 10x10 feature maps (if downsampled by 2)
    # flattened into a long input vector
    layer2 = FullyConnectedLayer(200, 10)
    print "initialized fully-connected layer"
    params = np.concatenate((layer0.W.flatten(), layer0.bias.flatten(), layer2.W.flatten(), layer2.bias.flatten()))
    velocity = np.zeros(params.shape)

    for i in range(0, epochs):
        correct_class = 0
        cost = 0.0
        # shuffle the dataset--shuffle_vec will be used as indices
        shuffle_vec = rand.permutation(data.shape[2])

        for j in range(0, data.shape[2] - minibatch_size + 1, minibatch_size):
            # perform gradient descent w/each batch
            it += 1

            if it == 20:
                # increase momentum after 20 iterations
                momentum = 0.9

            # gradient should be an unrolled vector of the avg. sum of the 256 gradients gotten
            # from the forward pass and backprop
            for k in range(0, minibatch_size):
                layer0.forwardprop(data[:,:,shuffle_vec[j+k]].reshape((28,28,1)))
                layer1.downsample(layer0.output, (20,20,2))
                layer2_input = layer1.output.flatten()
                layer2.softmax_output(layer2_input.reshape((layer2_input.size, 1)))
                cost += J(layer2.output, labels[shuffle_vec[j+k]])
                # print "%d %d" % (np.argmax(layer2.output), labels[shuffle_vec[j+k]])

                if np.argmax(layer2.output) == labels[shuffle_vec[j+k]]:
                    correct_class += 1

                # backprop
                layer2.backprop(0, 0, encode_label(labels[shuffle_vec[j+k]]))
                layer1.upsample(layer2, 0)
                layer0.backprop(layer1)
                # flatten the gradient vector
                if k == 0:
                    grad = np.concatenate((layer0.gradient_w.flatten(), layer0.gradient_b.flatten(), layer2.gradient_w.flatten(), layer2.gradient_b.flatten()))
                else:
                    grad += np.concatenate((layer0.gradient_w.flatten(), layer0.gradient_b.flatten(), layer2.gradient_w.flatten(), layer2.gradient_b.flatten()))

            grad /= minibatch_size
            # update velocity vector
            velocity = momentum*velocity + alpha*grad
            params =  params - velocity

            # update the parameters
            layer0.W = params[0:layer0.W.flatten().size].reshape(layer0.W.shape)
            next_begin = layer0.W.flatten().size
            layer0.bias = params[next_begin:next_begin+layer0.bias.flatten().size].reshape(layer0.bias.shape)
            next_begin += layer0.bias.flatten().size
            layer2.W = params[next_begin:next_begin+layer2.W.flatten().size].reshape(layer2.W.shape)
            next_begin += layer2.W.flatten().size
            layer2.bias = params[next_begin:].reshape(layer2.bias.shape)

        # reduce learning rate by half after each epoch
        alpha /= 2.0
        print "%d correct classifications" % correct_class
        print "cost function is ", cost/(minibatch_size*(data.shape[2] - minibatch_size + 1))
