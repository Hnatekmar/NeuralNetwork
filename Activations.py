import numpy as np

class Relu:
    def derivative(self, x):
        return np.maximum(0, x)

    def __call__(x):
        return np.maximum(0, x)

class Sigmoid:
    def derivative(self, x):
        return 0

    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))
