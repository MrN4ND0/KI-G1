class Activation_ReLU: # more Information for Rectified Linear Unit (https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Acitvation_Softmax: #more information for Softmax actvation (https://machinelearningmastery.com/softmax-activation-function-with-python/)
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
