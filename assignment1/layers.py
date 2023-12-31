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
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

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
        self.W.grad = np.dot((self.X).T, d_out)
        self.B.grad = np.sum(d_out, axis = 0).reshape(-1, (self.B.value).shape[1])

        return np.dot(d_out, (self.W.value).T)

    def params(self):
        return {'W': self.W, 'B': self.B}

    def clear_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
