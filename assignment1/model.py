import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.W1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU = ReLULayer()
        self.W2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.reg = reg
        self.hidden_layer_size = hidden_layer_size

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        self.W1.clear_grad()
        self.W2.clear_grad()
        loss, dY = softmax_with_cross_entropy(self.W2.forward(self.ReLU.forward(self.W1.forward(X))), y)
        self.W1.backward(self.ReLU.backward(self.W2.backward(dY)))
        l, grad = l2_regularization(self.W1.params()['W'].value, self.reg)
        loss += l
        self.W1.params()['W'].grad += grad
        l, grad = l2_regularization(self.W2.params()['W'].value, self.reg)
        loss += l
        self.W2.params()['W'].grad += grad
        l, grad = l2_regularization(self.W1.params()['B'].value, self.reg)
        loss += l
        self.W1.params()['B'].value += grad
        l, grad = l2_regularization(self.W2.params()['B'].value, self.reg)
        loss += l
        self.W2.params()['B'].value += grad
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        """
        return np.argmax(self.W2.forward(self.ReLU.forward(self.W1.forward(X))), axis = 1)

    def params(self):
        result = {'W1': self.W1.params()['W'], 'B1': self.W1.params()['B'], 'W2': self.W2.params()['W'], 'B2': self.W2.params()['B']}
        return result