import numpy as np
from testdatacreator import spiral_data
import math

#simple neuron network (NN) with 4 inputs and 3 neurons (weights + bias)
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output_simple = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print("simple: \n" + str(output_simple))

#more complex implementation of the same NN
inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights,biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print ("complex: \n" + str(layer_outputs))

#with numpy of the same NN
inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output_numpy = np.dot(weights, inputs) + biases #weights first input cause shape
print ("numpy: \n" + str(output_numpy))

#input from vector to batch (singlesample to multisample(good for generalisation))
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output_numpy = np.dot(inputs, np.array(weights).T) + biases #transpose the weights matrix to fit shape
print ("numpy_with_batch: \n" + str(output_numpy))

#adding a second layer
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


layer1_output_numpy = np.dot(inputs, np.array(weights1).T) + biases1 #transpose the weights matrix to fit shape

layer2_output_numpy = np.dot(layer1_output_numpy, np.array(weights2).T) + biases2

print ("numpy_with 2 layers: \n" + str(layer2_output_numpy))

#upgrading code

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5], #X = Input
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #creating random weights for layer (transpose not needed anymore)
        self.biases = np.zeros((1, n_neurons))  #creating random biases for layer
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print("upgraded_layer1: \n" + str(layer1.output))
layer2.forward(layer1.output)
print("upgraded_layer2: \n" + str(layer2.output))

#lineral activation
np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5], #X = Input
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []

for i in inputs:
    if i > 0:
        outputs.append(i)
    elif i <= 0:
        outputs.append(0)
    #or better: output .append(max(0, i))  
          
print("lineral activation: \n" + str(outputs))

#converting into class

X, y = spiral_data(100, 3)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
 
#drivers for Activation_ReLU        
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print ("~~~activation sample:~~~~~~~~~~~~~~~~~~~~~~~~")
print("layer output: \n" + str(layer1.output))
print("function output: \n" + str(activation1.output))

#softmax
layer_outputs = [4.8 , 1.21, 2.385]
E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)
    
print("exp_values: \n" + str(exp_values))

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)
    
print ("norm values: \n" + str(norm_values))
print ("norm values sum: \n" + str(sum (norm_values)))

#convert to numpy
layer_outputs = [4.8 , 1.21, 2.385]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values)

print ("norm values with numpy: \n" + str(norm_values))

#addin batch functionality
layer_outputs = [[4.8 , 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs) #works with batch

norm_values = exp_values / np.sum(exp_values,axis=1, keepdims=True) #fixed shape problems

print ("norm values batch with numpy: \n" + str(norm_values))

#softmax function class
class Acitvation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 =Acitvation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("softmax: \n" + str(activation2.output[:5]))

#loss in raw py
import math
 
softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] + 
        math.log(softmax_output[1])*target_output[1] +
        math.log(softmax_output[2])*target_output[2])

print("loss of softmax: \n" + str(loss))

#loss but cooler with numpy and batch of inputs (catergorical crossentropy)
softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4], 
                           [0.02, 0.9, 0.08]])
target_class = [0, 1, 1]

neg_log = -np.log(softmax_output[range(len(softmax_output)), target_class])

print("loss but cooler: \n" + str(neg_log))

avg_loss = np.mean(neg_log) #WATCHOUT log 0 = inf --> fucks up the mean; clip the val of 0 to sth very small but not 0

print("average loss: \n" + str(avg_loss))

#to class
class Loss: 
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss (class); \n" + str(loss))