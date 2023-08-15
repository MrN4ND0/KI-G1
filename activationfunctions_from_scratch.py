import numpy as np
import math
from testdatacreator import spiral_data
from layers import Layer_Dense

# 1. lineral activation --> more Information for Rectified Linear Unit (https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],  # X = Input
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []

for i in inputs:
    if i > 0:
        outputs.append(i)
    elif i <= 0:
        outputs.append(0)
    # optional output .append(max(0, i))

print("1.: \n" + str(outputs))

# 2. converting into class

X, y = spiral_data(100, 3)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# test drivers for Activation_ReLU
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print("2. layer output: \n" + str(layer1.output))
print("2. function output: \n" + str(activation1.output))

# 3. softmax --> more information for Softmax actvation (https://machinelearningmastery.com/softmax-activation-function-with-python/)
layer_outputs = [4.8, 1.21, 2.385]
E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E ** output)

print("3. exp_values: \n" + str(exp_values))

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print("3. norm values: \n" + str(norm_values))
print("3. norm values sum: \n" + str(sum(norm_values)))

# 4. convert to numpy
layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values)

print("4. norm values with numpy: \n" + str(norm_values))

# 5. adding batch functionality
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)  # works with batch

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # fixed shape problems

print("5. norm values batch with numpy: \n" + str(norm_values))


# 6. softmax function class
class Acitvation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# test driver for softmax
X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Acitvation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("6. softmax: \n" + str(activation2.output[:5]))