#Things we need to our project
import numpy as np
from testdatacreator import spiral_data
from layers import Layer_Dense
from activationfunctions import Activation_ReLU, Acitvation_Softmax
from losscalculator import Loss_CategoricalCrossentropy

# Creation of a test data set with X: x,y coordinates of a point and y: the class to which the point belongs.
X, y = spiral_data(100, 3)

# Creation of the input layer of the ANN. Here a fully connected layer with 2 input neurons (one for the x-coordinate and one for the y-coordinate) and the first 10 neurons of the hidden layer.
dense1 = Layer_Dense(2, 10)
activation1 = Activation_ReLU()

# Creation of the hidden layers of the ANN. Here, two fully connected and identical layers with 10 neurons per layer.
dense2 = Layer_Dense(10, 10)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(10, 10)
activation3 = Activation_ReLU()

# Creation of the output layer of the ANN. Here fully connected layer of 10 input neurons (form the last hidden layer) and 3 output neurons wich represent the classes of classification.
dense4 = Layer_Dense(10, 3)
activation4 = Acitvation_Softmax()

# Setting of the lossfunction.
loss_function = Loss_CategoricalCrossentropy()

# Creation of a variables which store weights and bias values for the matter of storing the learning progression.
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
best_dense3_weights = dense3.weights.copy()
best_dense3_biases = dense3.biases.copy()
best_dense4_weights = dense4.weights.copy()
best_dense4_biases = dense4.biases.copy()

# Creation of an optimizer which alters the weights and biases by a random amount. Runs for 1.000.000 iterations.
for iteration in range(10000):
    dense1.weights += 0.05 * np.random.randn(2, 10)
    dense1.biases += 0.05 * np.random.randn(1, 10)
    dense2.weights += 0.05 * np.random.randn(10, 10)
    dense2.biases += 0.05 * np.random.randn(1, 10)
    dense3.weights += 0.05 * np.random.randn(10, 10)
    dense3.biases += 0.05 * np.random.randn(1, 10)
    dense4.weights += 0.05 * np.random.randn(10, 3)
    dense4.biases += 0.05 * np.random.randn(1, 3)
 
    # Main driver code for passing values through the network of neurons and activation functions.
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    dense4.forward(activation3.output)
    activation4.forward(dense4.output)

    # Calculating metric for the current run by evaluation of the outputs with our predictions.
    loss = loss_function.calculate(activation4.output, y)

    prediction = np.argmax(activation4.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(prediction == y)

    # Better runs get stored in our variables. Even or worse runs get ignored.
    if loss < lowest_loss:
        print('New weights found , iteration: ', iteration, 'loss: ', loss, 'accuracy: ', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        best_dense3_weights = dense3.weights.copy()
        best_dense3_biases = dense3.biases.copy()
        best_dense4_weights = dense4.weights.copy()
        best_dense4_biases = dense4.biases.copy()
       
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        dense3.weights = best_dense3_weights.copy()
        dense3.biases = best_dense3_biases.copy()
        dense4.weights = best_dense4_weights.copy()
        dense4.biases = best_dense4_biases.copy()

# Output of the final values of the weights and biases.
print('[Dense1] Weights: ', dense1.weights, '\n', '[Dense1] Biases: ', dense1.biases)
print('[Dense2] Weights: ', dense2.weights, '\n', '[Dense2] Biases: ', dense2.biases)
print('[Dense3] Weights: ', dense3.weights, '\n', '[Dense3] Biases: ',  dense3.biases)
print('[Dense4] Weights: ', dense4.weights, '\n', '[Dense4] Biases: ', dense4.biases)


