import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig

class ConvLayer():
    """Perform the convolution and apply the sigmoid activation for a layer, then perform pooling"""

    def __init__(self, input, prev_layer_shape, filter_shape, pooling_x, pooling_y):
        """
        :type input: 3-d matrix at the moment
        :param input: the input to the layer--either a 2-D matrix or set of 2-D feature maps

        :type prev_layer_shape: tuple
        :param prev_layer_shape: (num input feature maps, height, width) of the previous layer

        :type filter_shape: tuple
        :param filter_shape: (num filters, num input feature maps, height, width)

        :type pooling_x: int
        :param pooling_x: Determine num of rows in the max-pooling matrix

        :type pooling_y: int
        :param pooling_y: Determine num of columns in the max-pooling matrix
        """

        self.input = input

        # instantiate weight matrix
        rng = rand.RandomState(23455)
        W = np.asarray(rng.uniform(low=-1.0/(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    high=1.0/(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    size=filter_shape
        ))

        # instantiate bias. need a bias for each filter
        bias_shape = (filter_shape[0],)
        bias = np.asarray(rng.uniform(low=-0.5, high=-0.5, size=bias_shape))

        # perform the convolution, each feature map with its respective layer,
        # sum the resultant convolution layers together
        conv_output = np.zeros((prev_layer_shape[2]-filter_shape[2]+1, prev_layer_shape[3]-filter_shape[3]+1, filter_shape[0]))
        for i in range(0, filter_shape[0]):
            for j in range(0, filter_shape[1]):
                #rotation keeps the convolution matrix oriented correctly
                conv_output[:,:,i] += sig.convolve2d(input[j,:,:], np.rot90(W[j,i,:,:], 2), mode='valid')
            #add bias for this filter
            conv_output[:,:,i] += conv_output[:,:,i] + bias[i]

        #apply sigmoid
        output = np.tanh(conv_output)
        self.conv_output = conv_output

        # output is a 3-D matrix, essentially a 'stacking' of the different feature maps produced
        self.output = output
        self.W = W
        self.bias = bias
        self.params = [W, bias]

    def backprop(self, layer):
        # previous layer should always be a pooling layer--the layer should have upsample() called
        self.error = np.zeros(output.shape)
        for i in range(0, layer.output.shape[2]):
            # for each filter do the following
            self.error[:,:,i] = np.multiply(layer.error, (1 - np.multiply(layer.error, layer.error)))
        # now to calculate gradients
        for i in range(0, self.filter_shape[0]):
            for j in range(0, self.filter_shape[1]):
                if j == 0:
                    self.gradient_w[:,:,i] = sig.convolve2d(input[j,:,:], np.rot90(self.error[:,:,i], 2), mode='valid')
                else:
                    self.gradient_w[:,:,i] += sig.convolve2d(input[j,:,:], np.rot90(self.error[:,:,i], 2), mode='valid')
        # may not be correct form of the bias gradient
        self.gradient_b = np.array(i)
        for i in range(0, self.filter_shape[0]):
            self.gradient_b[i] = np.sum(self.error[:,:,i])
        
