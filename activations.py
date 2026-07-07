import numpy as np

class Relu:
    def derivative(self, x):
        result = self(x)
        result[result > 0] = 1
        return result

    def __call__(self, x):
        return np.maximum(0, x)

class Sigmoid:
    def derivative(self, x):
        sigmoid = self(x)
        return (1.0 - sigmoid) * sigmoid

    def __call__(self, x):
        # Clip to prevent numerical overflow in exp
        return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
