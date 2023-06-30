import numpy as np


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
    '''
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
    '''
    assert predictions.ndim in (1, 2)
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)
    if (dprediction.ndim == 1):
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(dprediction.shape[0]), target_index.flatten()] -= 1
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = np.sum(W ** 2) * reg_strength
    grad = W * 2 * reg_strength
    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W
    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes
    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss
    '''
    loss, dY = softmax_with_cross_entropy(np.dot(X, W), target_index)
    dW = np.dot(X.T, dY)
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    return loss, dW

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            l = 0
            for batch in batches_indices:
                loss, dW = linear_softmax(X[batch], self.W, y[batch])
                l2_loss, grad = l2_regularization(self.W, reg)
                self.W -= (dW + grad)*learning_rate
                l += loss + l2_loss
            l /= len(batches_indices)
            print("Epoch %i, loss: %f" % (epoch, l))
            loss_history.append(l)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")

            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        return np.argmax(np.dot(X, self.W), axis = 1)

    def copy_W(self):
        Wrep = self.W.copy()
        return Wrep






