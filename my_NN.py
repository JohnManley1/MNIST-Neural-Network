import numpy as np
from numpy import random as npr

   
# Layer or a collection of nodes in a neural network
class Layer:

    def __init__(self, n_nodes, n_inputs, activation, learning_rate=0.01):
        # create collection of nodes biases and weights and set variables
        self.W = npr.randn(n_nodes, n_inputs) * 0.01  # weights for each node
        self.b = npr.randn(n_nodes, 1) * 0.01  # biases for each node
        self.activation = activation
        self.learning_rate = learning_rate
        # record inputs and outputs through the layer
        self.A_prev = None  # inputs to the layer
        self.Z = None # raw outputs
        self.A = None  # activated outputs
        # record variable gradients
        self.dW = np.zeros((n_nodes, n_inputs))  # gradients for weights
        self.db = np.zeros((n_nodes, 1))  # gradients for biases
    
    def forward(self, X):
        self.A_prev = X
        # apply forward to each node using dot product
        self.Z =  np.dot(self.W, X) + self.b # raw outputs
        # apply activation to outputs
        self.A = self.activation(self.Z)
        return self.A
    
    def backprop(self, delta_next=None, W_next=None, y=None):
        # compute delta based on if layer is output layer or not
        if y is not None: # output layer
            # compute delta based on layer activation
            if isinstance(self.activation, Softmax):
                delta = (self.A - y)
            else:
                delta = (self.A - y) * self.activation.derivative(self.Z)
        else: # hidden layer
            delta = np.dot(W_next.T, delta_next) * self.activation.derivative(self.Z)
        
        # compute gradients for weights and bias
        self.dW += np.dot(delta, self.A_prev.T)
        self.db += delta

        return self.W, delta
    
    def update(self, batch_size):
        # update weights and biases
        self.W -= self.dW * (self.learning_rate / batch_size)
        self.b -= self.db * (self.learning_rate / batch_size)
        # reset gradients for next iteration
        self.dW = np.zeros_like(self.dW)
        self.db = np.zeros_like(self.db)


# Neural Network conatining multiple layers
class Neural_Network:

    def __init__(self, n_inputs, n_outputs, hidden_layers, hidden_size, hidden_activation, output_activation, learning_rate=0.01):
        # add first hidden layer
        self.layers = [Layer(hidden_size, n_inputs, hidden_activation, learning_rate)]
        # add the remaining hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(Layer(hidden_size, hidden_size, hidden_activation, learning_rate))
        # add the ouput layer used for predictions
        self.layers.append(Layer(n_outputs, hidden_size, output_activation, learning_rate))
    
    def forward(self, X):
        # apply forward pass through each layer
        for layer in self.layers:
            X = layer.forward(X)
        # return ouput layers results
        return X
    
    def backprop(self, y):
        # start with output layer
        W, delta = self.layers[-1].backprop(y=y)
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            W, delta = layer.backprop(delta_next=delta, W_next=W)
    
    def train(self, batches, outputs):
        # iterate through each batch and expected output
        for X, y in zip(batches, outputs):
            # forward pass
            self.forward(X)
            # backpropagation
            self.backprop(y)
        # update weights and biases
        self.update(len(batches))
        
    def update(self, batch_size):
        # update weights and biases for each layer
        for layer in self.layers:
            layer.update(batch_size)
    
    def test(self, batches, ouputs):
        # choose loss function based on output layer activation
        loss = 0
        loss_function = cross_entropy if isinstance(self.layers[-1].activation, Softmax) else MSE
        # iterate through each batch and calculate loss
        for X, y in zip(batches, ouputs):
            loss += loss_function(y, self.forward(X))
        # return average loss over all batches
        return loss / len(batches)
        





# Activation functions and derivatives

class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Leaky_ReLU:
    def __call__(self, x):
        return np.where(x > 0, x, 0.01 * x)
    def derivative(self, x):
        return np.where(x > 0, 1, 0.01)

class Tanh:
    def __call__(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sig = self.__call__(x)
        return sig * (1 - sig)


class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return e_x / e_x.sum(axis=0)

    def derivative(self, x):
        # simplified derivative for softmax
        s = self.__call__(x)
        return s * (1 - s)  



# Loss functions

# Mean Squared Error
def MSE(expected, actual):
    sq_dif = (expected - actual) ** 2
    return np.sum(sq_dif)

# Cross Entropy
def cross_entropy(expected, actual):
    # avoid log(0) by adding a small value
    d = 1e-15
    actual = np.clip(actual, d, 1 - d)
    return -np.sum(expected * np.log(actual))
