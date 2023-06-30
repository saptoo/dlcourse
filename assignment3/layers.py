import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    return np.sum(W ** 2) * reg_strength, W * 2 * reg_strength # loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    pr = predictions.copy()
    if (predictions.ndim == 1) :
        pr -= np.max(pr)
        pr = np.exp(pr)
        pr /= np.sum(pr)
    else:
        pr -= (np.max(pr, axis = 1)).reshape(-1, 1)
        pr = np.exp(pr)
        pr /= (np.sum(pr, axis = 1)).reshape(-1, 1)
    return pr

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    assert probs.ndim in (1, 2)
    if probs.ndim == 1:
        return - np.log(probs[target_index])
    else:
        return - np.sum(np.log(probs[np.arange(probs.shape[0]), target_index.flatten()]))
    # Your final implementation shouldn't have any loops

def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    assert predictions.ndim in (1, 2)
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)
    if (dprediction.ndim == 1):
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(dprediction.shape[0]), target_index.flatten()] -= 1
    return loss, dprediction

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        # возможно придется переписать чтоб не менялся X
        self.X = X > 0.
        return np.maximum(0., X)

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_out_copy = d_out.copy()
        d_out_copy[np.logical_not(self.X)] = 0
        return d_out_copy

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

    def clear_grad(self):
        pass


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0).reshape(-1, self.B.value.shape[1])

        return np.dot(d_out, self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}

    def clear_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X.copy()

        if self.padding:
            self.X = np.zeros((batch_size,
                               height + 2 * self.padding,
                               width + 2 * self.padding,
                               channels), dtype=X.dtype)
            self.X[:, self.padding: -self.padding, self.padding: -self.padding, :] = X

        _, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        output = np.zeros((batch_size, out_height, out_width, self.out_channels), dtype=X.dtype)

        w_reshape = self.W.value.reshape((self.filter_size ** 2 * self.in_channels, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                x_window = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]\
                    .reshape((batch_size, self.filter_size ** 2 * channels))

                output[:, y, x, :] = np.dot(x_window, w_reshape)
        output += self.B.value
        return output

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        output = np.zeros_like(self.X)
        # Try to avoid having any other loops here too
        w_reshape = self.W.value.reshape((self.filter_size ** 2 * self.in_channels, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                x_window = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :].reshape((batch_size, self.filter_size ** 2 * channels))

                d = d_out[:, y, x, :]

                self.W.grad += np.dot(x_window.T, d.reshape((batch_size, out_channels)))\
                    .reshape((self.filter_size, self.filter_size, self.in_channels, self.out_channels))

                output[:, y: y + self.filter_size, x: x + self.filter_size, :] += \
                    np.dot(d.reshape((batch_size, out_channels)), w_reshape.T)\
                    .reshape((batch_size, self.filter_size, self.filter_size, self.in_channels))

        #  if self.padding:
        #      tmp = np.zeros((batch_size, height, width, out_channels), dtype=float)
        #      tmp[:, self.padding: -self.padding, self.padding: -self.padding, :] = output
        #      output = tmp
        self.B.grad = np.sum(d_out.reshape(batch_size, -1), axis=0)
        if self.padding:
            return output[:, self.padding: -self.padding, self.padding: -self.padding, :]
        return output

    def params(self):
        return {'W': self.W, 'B': self.B}

    def clear_grad(self):
        self.W.grad = np.zeros_like(self.W.grad)
        self.B.grad = np.zeros_like(self.B.grad)


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, channels = X.shape

        assert (height - self.pool_size) % self.stride == 0
        assert (width - self.pool_size) % self.stride == 0

        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1

        output = np.zeros((batch_size, out_height, out_width, channels), dtype=float)
        y1 = 0
        for y in range(out_height):
            x1 = 0
            for x in range(out_width):
                output[:, y, x, :] = \
                    np.max(X[:, y1: y1 + self.pool_size, x1: x1 + self.pool_size, :], axis=(1, 2))
                x1 += self.stride
            y1 += self.stride
        return output

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, channels = d_out.shape
        output = np.zeros_like(self.X, dtype=float)
        for i in range(batch_size):
            y1 = 0
            for y in range(out_height):
                x1 = 0
                for x in range(out_width):
                    for j in range(channels):
                        idx = np.argmax(self.X[i, x1: x1 + self.pool_size, y1: y1 + self.pool_size, j])
                        output[i, y1: y1 + self.pool_size,
                        x1: x1 + self.pool_size, j][idx // self.pool_size, idx % self.pool_size] += d_out[i, y, x, j]
                    x1 += self.stride
                y1 += self.stride

        return output

    def params(self):
        return {}

    def clear_grad(self):
        pass

class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        # Layer should return array with dimensions
        # [batch_size, hight * width * channels]
        return X.reshape((self.X_shape[0], -1))

    def backward(self, d_out):
        return d_out.reshape((self.X_shape[0], self.X_shape[1], self.X_shape[2], self.X_shape[3]))

    def params(self):
        # No params!
        return {}

    def clear_grad(self):
        pass
