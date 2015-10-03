import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig
import scipy.special as spec

class ConvLayer():
    """Perform the convolution and apply the sigmoid activation for a layer, then perform pooling"""

    def __init__(self, prev_layer_shape, filter_shape):
        """
        :type prev_layer_shape: tuple
        :param prev_layer_shape: (height, width, num feature maps) of the previous layer

        :type filter_shape: tuple
        :param filter_shape: (height, width, num filters)
        """

        self.prev_layer_shape = prev_layer_shape
        self.filter_shape = filter_shape
        # instantiate weight matrix
        rng = rand.RandomState(23455)
        # W = rand.randint(10, size=filter_shape)
        W = np.asarray(rng.uniform(low=-1.0/np.sqrt(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    high=1.0/np.sqrt(filter_shape[0]*filter_shape[1]*filter_shape[2]),
                                    size=filter_shape
        ))

        # instantiate bias. need a bias for each filter
        bias_shape = (filter_shape[2],)
        bias = np.zeros(bias_shape)
        # bias = np.asarray(rng.uniform(low=-0.5, high=-0.5, size=bias_shape))

        self.W = W
        self.bias = bias
        self.bias_shape = bias_shape
        self.params = [W, bias]

    # perform forward propagation--generate the output
    def forwardprop(self, input):
        """
        :type input: 3-d matrix at the moment
        :param input: the input to the layer--either a 2-D matrix or set of 2-D feature maps
        """
        # perform the convolution, each feature map with its respective layer,
        # sum the resultant convolution layers together
        conv_output = np.zeros((self.prev_layer_shape[0]-self.filter_shape[0]+1, self.prev_layer_shape[1]-self.filter_shape[1]+1, self.filter_shape[2]))
        for i in range(0, self.filter_shape[2]):
            for j in range(0, self.prev_layer_shape[2]):
                #rotation keeps the convolution matrix oriented correctly
                conv_output[:,:,i] += sig.convolve2d(input[:,:,j], np.rot90(self.W[:,:,i], 2), mode='valid')
            #add bias for this filter
            conv_output[:,:,i] = conv_output[:,:,i] + self.bias[i]

        self.input = input
        self.conv_output = conv_output

        # max-pooling operation
        output = np.zeros((conv_output.shape[0]/2, conv_output.shape[1]/2, conv_output.shape[2]))
        max_indices = np.zeros((conv_output.shape[0]/2, conv_output.shape[1]/2, conv_output.shape[2]))
        for k in range(0, conv_output.shape[2]):
            for i in range(0, conv_output.shape[0]/2):
                for j in range(0, conv_output.shape[1]/2):
                    output[i,j,k] = np.amax(conv_output[2*i:2*i+2, 2*j:2*j+2, k])
                    max_indices[i,j,k] = np.argmax(conv_output[2*i:2*i+2, 2*j:2*j+2,k])

        self.max_indices = max_indices

        # apply sigmoid
        output = np.tanh(output)
        # output is a 3-D matrix, essentially a 'stacking' of the different feature maps produced and downsampled
        self.output = output

    def backprop(self, layer, flag):
        if flag == 0:
            # previous layer was a fully connected layer, so output was flattened into a vector
            # first determine the partials for the vector components, then re-map into feature maps
            vec_error = np.dot(np.transpose(layer.W), layer.error)
            self.vec_error = vec_error
            self.error = np.zeros(self.conv_output.shape)
            # map back to feature maps
            counter = -1
            for i in range(0, self.output.shape[0]):
                for j in range(0, self.output.shape[1]):
                    for k in range(0, self.output.shape[2]):
                        counter += 1
                        # generalize this mapping later
                        # differentiate the sigmoid here too
                        delt = vec_error[counter]*(1.0-self.output[i,j,k]*self.output[i,j,k])
                        if self.max_indices[i,j,k] == 0:
                            self.error[2*i,2*j,k] = delt
                        elif self.max_indices[i,j,k] == 1:
                            self.error[2*i,2*j+1,k] = delt
                        elif self.max_indices[i,j,k] == 2:
                            self.error[2*i+1,2*j,k] = delt
                        else:
                            self.error[2*i+1,2*j+1,k] = delt
        else:
            # previous layer was another convolutional layer--no need to re-map into feature maps
            # will need to figure out how to get the necessary gradients and errors, though
            # no weights applied, so no weights come into the backprop?

            # identify no. of weights in the previous convolutional layer
            num_maps = layer.filter_shape[2]
            self.error = np.zeros(self.conv_output.shape)
            s_error = np.zeros(self.output.shape)
            for i in range(s_error.shape[2]):
                for j in range(layer.error.shape[2]):
                    s_error[:,:,i] += sig.convolve2d(layer.error[:,:,j], layer.W[:,:,j], mode='full')
                    print s_error[:,:,i]        

            s_error = np.multiply(s_error, 1.0 - np.multiply(self.output, self.output))
            self.s_error = s_error
            for i in range(0, self.output.shape[0]):
                for j in range(0, self.output.shape[1]):
                    for k in range(0, self.output.shape[2]):
                        # generalize mapping later
                        if self.max_indices[i,j,k] == 0:
                            self.error[2*i,2*j,k] = s_error[i,j,k]
                        elif self.max_indices[i,j,k] == 1:
                            self.error[2*i,2*j+1,k] = s_error[i,j,k]
                        elif self.max_indices[i,j,k] == 2:
                            self.error[2*i+1,2*j,k] = s_error[i,j,k]
                        else:
                            self.error[2*i+1,2*j+1,k] = s_error[i,j,k]

        # now to calculate gradients
        self.gradient_w = np.zeros((self.filter_shape[0], self.filter_shape[1], self.filter_shape[2]))
        for i in range(0, self.filter_shape[2]):
            for j in range(0, self.prev_layer_shape[2]):
                # if j == 0:
                #     self.gradient_w[:,:,i] = sig.convolve2d(self.input[:,:,j], np.rot90(self.error[:,:,i], 2), mode='valid')
                # else:
                self.gradient_w[:,:,i] += sig.convolve2d(self.input[:,:,j], np.rot90(self.error[:,:,i], 2), mode='valid')

        self.gradient_b = np.zeros(self.filter_shape[2])
        for i in range(0, self.filter_shape[2]):
            self.gradient_b[i] = np.sum(self.error[:,:,i])
