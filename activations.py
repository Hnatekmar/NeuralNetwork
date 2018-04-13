import numpy as np

class Relu:
    def derivative(self, x):
        result = self(x)
        result[result > 0] = 1
        return result

    def __call__(x):
        return np.maximum(0, x)

class Sigmoid:
    EPSILON = 1e-25
    def derivative(self, x):
        sigmoid = self(x)
        return (1.0 - sigmoid) * sigmoid

    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x + EPSILON))
