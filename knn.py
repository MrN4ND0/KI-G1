from testdatacreator import spiral_data
from main import Layer_Dense, Activation_ReLU, Acitvation_Softmax, Loss, Loss_CategoricalCrossentropy

X, y = spiral_data(100, 3)

#input layer
dense1 = Layer_Dense(2, 10)
activation1 = Activation_ReLU()

#inner layers
dense2 = Layer_Dense(10, 10)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(10, 10)
activation3 = Activation_ReLU()

dense4 =Layer_Dense(10, 10)
activation4 =Activation_ReLU()

#output layer
dense5 = Layer_Dense(10, 3)
activation5 = Acitvation_Softmax()

#driver
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

print(activation5.output)