import numpy as np
import math
from layers import Layer_Dense
from activationfunctions_from_scratch import Activation_ReLU, Acitvation_Softmax
from testdatacreator import spiral_data

# 1. calculation of loss
softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print("1. loss of softmax: \n" + str(loss))

#2.  loss but cooler with numpy and batch of inputs (catergorical crossentropy)
softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
target_class = [0, 1, 1]

neg_log = -np.log(softmax_output[range(len(softmax_output)), target_class])

print("2. loss but cooler: \n" + str(neg_log))

avg_loss = np.mean(neg_log)  # WATCHOUT log 0 = inf --> destroys the mean; clip the val of 0 to sth very small but not 0 (line 40)

print("2. average loss: \n" + str(avg_loss))


#3. to class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

# test driver for CategoricalCrossentropy
X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Acitvation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("3. Loss (class); \n" + str(loss))