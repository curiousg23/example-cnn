import numpy as np
from numpy import random as rand
import scipy.sparse as sparse
import scipy.signal as sig

class FullyConnectedLayer():
    """Fully conntected layer that takes in inputs (either from another fully-connected layer
    or the last pooling layer--will flatten the feature maps into one giant vector)
    A sigmoid activation function is used"""

    def __init__(self, input, f_in, f_out):
        """
        :type input: vector(?) or matrix with (length of flattened feature vector, num of inputs)
        :param input: the input to the fully-connected layer--likely 1-D

        :type f_in: int
        :param f_in: the number of inputs--links from the previous layer

        :type f_out: int
        :param f_out: the number of outputs--inputs to the next layer
        """

        self.input = input

        # weight matrix initialization
        rng = rand.RandomState(23455)
        W = np.asarray(rng.uniform(low=-1.0/(np.sqrt(f_in)), high=1.0/(np.sqrt(f_in)), size=(f_out,f_in)))

        #bias initialization
        bias = np.zeros((f_out,1)) #potentially need the 2nd dimension to be nonsingular if multiple batches

        self.W = W
        self.bias = bias

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def softmax(self, x):
        denom = 0
        for i in range(0, x.size):
            denom += np.exp(x[i])
        return np.exp(x)/denom

    def forwardprop(self, input):
        #perform activation
        lin_output = np.dot(self.W, input) + self.bias
        output = self.sigmoid(lin_output)
        self.output = output
        self.params = [self.W, self.bias]
        self.lin_output = lin_output

    # perform output activation with softmax
    def softmax_output(self, input):
        lin_output = np.dot(self.W, input) + self.bias
        output = self.softmax(lin_output)
        self.output = output
        self.params = [self.W, self.bias]
        self.lin_output = lin_output

    def backprop(self, layer, flag, target):
        if flag == 0:
            # softmax layer instead of sigmoid, error function
            output = self.softmax(self.lin_output)
            self.error = output - target
            print self.error.shape
            print np.transpose(self.input).shape
            self.gradient_w = np.dot(self.error, np.transpose(self.input))
            self.gradient_b = self.error
        else:
            # sigmoid layer connected to another layer after it--these may not be correct, will need to calculate
            self.error = np.multiply(np.transpose(layer.W)*layer.error, np.multiply(self.output, 1-self.output)) #derivative of sigmoid is sigma*(1-sigma)
            self.gradient_w = np.dot(self.error, np.transpose(self.input))
            self.gradient_b = self.error
