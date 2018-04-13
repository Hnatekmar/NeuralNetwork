import layer
import numpy as np

class Classic(layer.ILayer):
    def __init__(self, size, activation):
        super(size)
        self.weights = np.random.rand(size[0], size[1])
        self.activation = activation

    def forward(self, x):
        return x * self.weights
