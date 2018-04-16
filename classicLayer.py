import layer
import numpy as np
import random

class Classic(layer.ILayer):
    def __init__(self, size, activation):
        layer.ILayer.__init__(self, size)
        self.weights = np.random.rand(size[0] + 1, size[1]) * 2.0 - 1.0
        self.weights[0, :] = 1 # Bias
        self.activation = activation

    def forward(self, x):
        x = np.append(x, np.ones((x.shape[0], 1)), 1)
        return self.activation(x @ self.weights)

    def randomNeighbor(self):
        alpha = 0.1
        shape = self.weights.shape
        self.weights[random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1)] += (random.random() * 2.0 - 1.0) * alpha

