import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #creating random weights for layer (transpose not needed anymore)
        self.biases = np.zeros((1, n_neurons))  #creating random biases for layer
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases