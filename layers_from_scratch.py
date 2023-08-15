import numpy as np

# 1. simple neuron network (NN) with 4 inputs and 3 neurons (weights + bias)
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output_simple = [
    inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
    inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
    inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]
print("1.: \n" + str(output_simple))

# 2. more complex implementation of the same NN(1.)
inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print("2.: \n" + str(layer_outputs))

# 3. with numpy of the same NN(1. & 2.)
inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output_numpy = np.dot(weights, inputs) + biases  # weights first input cause shape
print("3.: \n" + str(output_numpy))

# 4. input from vector to batch (singlesample to multisample(good for generalisation))
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output_numpy = np.dot(inputs, np.array(weights).T) + biases  # transpose the weights matrix to fit shape
print("4.: \n" + str(output_numpy))

# 5. adding a second layer
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2.0, 3.0, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1.0, 2.0, -0.5]

layer1_output_numpy = np.dot(inputs, np.array(weights1).T) + biases1  # transpose the weights matrix to fit shape

layer2_output_numpy = np.dot(layer1_output_numpy, np.array(weights2).T) + biases2

print("5.: \n" + str(layer2_output_numpy))

# 6. upgrading code to dedicated class with functions

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],  # X = Input
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,
                                              n_neurons)  # creating random weights for layer (transpose not needed anymore)
        self.biases = np.zeros((1, n_neurons))  # creating random biases for layer

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print("6.: \n" + str(layer1.output))
layer2.forward(layer1.output)
print("6.: \n" + str(layer2.output))