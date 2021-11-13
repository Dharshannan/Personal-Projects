import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

# Revision stuff down below
'''ytrue = np.array([0, 1, 1])
M0 = np.array([[1, 2, 3], [2, 5, -1], [-1.5, 2.7, -0.8]])   #inputs
M1 = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
M2 = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
M3 = np.array([[0.21, -0.07, -0.14], [-0.07, 0.09, -0.02], [-0.14, -0.02, 0.16]])
out = (-M2/M1)/3
M4 = M3.copy()
M4[M0 <= 0] = 0
inputs = np.empty_like(out)
for index, (sigout, sigd) in enumerate(zip(M1, out)):
    #print(index, sigout, sigd)
    sigout = sigout.reshape(-1, 1)
    #print(sigout.shape)
    jacob = np.diagflat(sigout) - np.dot(sigout, sigout.T)
    #print(index, jacob, sigd, np.dot(jacob, sigd))
    inputs[index] = np.dot(jacob, sigd)

#print(inputs)
#print(((np.array([1, 2, 3])).reshape(-1, 1)).shape)
#print(np.sum(M1, axis=0, keepdims=True))
#print(np.maximum(0, M3))
#print(M4)
#print((np.sum(M0, axis=1, keepdims=True)).shape)
#print(M1*M2)
#print(np.sum(M1*M2, axis=1))
#print(np.eye(len(M1))[ytrue])'''


# Neural Network from scratch:

class Layers_Dense():
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1w = np.ones_like(self.weights)
            dL1w[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1w
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1b = np.ones_like(self.biases)
            dL1b[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1b
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_Relu():
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax():
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_outputs, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_outputs = single_outputs.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_outputs) - np.dot(single_outputs, single_outputs.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss():
    def regularization_loss(self, layer):
        regularization_loss = 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def calculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Categorical_Crossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

    def backward(self, dvalues, y_true):
        sample = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / sample

class Activation_Softmax_Loss_Categorical_Crossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Crossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        return self.loss.calculate(self.outputs, y_true)

    def backward(self, dvalues, y_true):
        sample = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(sample), y_true] -= 1
        self.dinputs = self.dinputs / sample

class Optimizer_SGD():
    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1. / (1. + self.decay*self.iterations))

    def update_params(self, layers):
        if self.momentum:
            if not hasattr(layers, 'weight_momentums'):
                layers.weight_momentums = np.zeros_like(layers.weights)
                layers.bias_momentums = np.zeros_like(layers.biases)

            weight_updates = self.momentum * layers.weight_momentums - self.current_learning_rate * layers.dweights
            layers.weight_momentums = weight_updates

            bias_updates = self.momentum * layers.bias_momentums - self.current_learning_rate * layers.dbiases
            layers.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layers.dweights
            bias_updates = -self.current_learning_rate * layers.dbiases

        layers.weights += weight_updates
        layers.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam():
    def __init__(self, learning_rate=0.001, decay=0, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1. / (1. + self.decay*self.iterations))

    def update_params(self, layers):
        if not hasattr(layers, 'weight_momentums'):
            layers.weight_momentums = np.zeros_like(layers.weights)
            layers.bias_momentums = np.zeros_like(layers.biases)
            layers.weight_cache = np.zeros_like(layers.weights)
            layers.bias_cache = np.zeros_like(layers.biases)

        layers.weight_momentums = self.beta1 * layers.weight_momentums + (1-self.beta1)*layers.dweights
        layers.bias_momentums = self.beta1 * layers.bias_momentums + (1 - self.beta1) * layers.dbiases
        layers.weight_cache = self.beta2 * layers.weight_cache + (1 - self.beta2) * layers.dweights ** 2
        layers.bias_cache = self.beta2 * layers.bias_cache + (1 - self.beta2) * layers.dbiases ** 2

        weight_momentums_corrected = layers.weight_momentums / (1 - self.beta1**(self.iterations + 1))
        bias_momentums_corrected = layers.bias_momentums / (1 - self.beta1**(self.iterations + 1))
        weight_cache_corrected = layers.weight_cache / (1 - self.beta2**(self.iterations + 1))
        bias_cache_corrected = layers.bias_cache / (1 - self.beta2**(self.iterations + 1))

        layers.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layers.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Layer_Dropout():
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=(inputs.shape)) / self.rate
        self.outputs = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# Define dataset:

X, y = spiral_data(samples=1000, classes=3)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
#plt.show()

dense1 = Layers_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_Relu()
dense2 = Layers_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_Categorical_Crossentropy()
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)
dropout1 = Layer_Dropout(0.5)

for epochs in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dropout1.forward(activation1.outputs)
    dense2.forward(dropout1.outputs)
    loss = loss_activation.forward(dense2.outputs, y)

    predictions = np.argmax(loss_activation.outputs, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epochs % 100:
        print(f'epoch: {epochs}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.outputs, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.outputs)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.outputs)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.outputs, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.outputs, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

