import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.model = [
            ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(2, 2),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(2, 2),
            Flattener(),
            FullyConnectedLayer(int(input_shape[0] / 4) ** 2 * conv2_channels, n_output_classes)]

    def temp_f(self, X):
        tmp = X.copy()
        for layer in self.model:
            tmp = layer.forward(tmp)
        return tmp

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for layer in self.model:
            layer.clear_grad()
        pred = self.temp_f(X)
        loss, dy = softmax_with_cross_entropy(pred, y)
        for layer in reversed(self.model):
            dy = layer.backward(dy)
        return loss


    def predict(self, X):
        return np.argmax(self.temp_f(X), axis=1)


    def params(self):
        result = {}

        for layer in self.model:
            for key, value in layer.params().items():
                result[key] = value

        return result
