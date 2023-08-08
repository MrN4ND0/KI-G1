import random

import numpy as np
from testdatacreator import spiral_data
from main import Layer_Dense, Activation_ReLU, Acitvation_Softmax, Loss, Loss_CategoricalCrossentropy

X, y = spiral_data(100, 3)

# input layer
dense1 = Layer_Dense(2, 10)
activation1 = Activation_ReLU()

# inner layers
dense2 = Layer_Dense(10, 10)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(10, 10)
activation3 = Activation_ReLU()

dense4 = Layer_Dense(10, 10)
activation4 = Activation_ReLU()

# output layer
dense5 = Layer_Dense(10, 3)
activation5 = Acitvation_Softmax()

# loss
loss_function = Loss_CategoricalCrossentropy()

# persistence layer
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
best_dense3_weights = dense3.weights.copy()
best_dense3_biases = dense3.biases.copy()
best_dense4_weights = dense4.weights.copy()
best_dense4_biases = dense4.biases.copy()
best_dense5_weights = dense5.weights.copy()
best_dense5_biases = dense5.biases.copy()

# optimizer
for iteration in range(10000):
    dense1.weights += 0.05 * np.random.randn(2, 10)
    dense1.biases += 0.05 * np.random.randn(1, 10)
    dense2.weights += 0.05 * np.random.randn(10, 10)
    dense2.biases += 0.05 * np.random.randn(1, 10)
    dense3.weights += 0.05 * np.random.randn(10, 10)
    dense3.biases += 0.05 * np.random.randn(1, 10)
    dense4.weights += 0.05 * np.random.randn(10, 10)
    dense4.biases += 0.05 * np.random.randn(1, 10)
    dense5.weights += 0.05 * np.random.randn(10, 3)
    dense5.biases += 0.05 * np.random.randn(1, 3)

    # driver
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    dense4.forward(activation3.output)
    activation4.forward(dense4.output)

    dense5.forward(activation4.output)
    activation5.forward(dense5.output)

    loss = loss_function.calculate(activation5.output, y)

    prediction = np.argmax(activation5.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(prediction == y)

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
        best_dense5_weights = dense5.weights.copy()
        best_dense5_biases = dense5.biases.copy()
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
        dense5.weights = best_dense5_weights.copy()
        dense5.biases = best_dense5_biases.copy()

