import numpy as np

from Layer import layerCreator

class NeuralNetwork:
    def __init__(self, architecture):
        self.layers = list(map(layerCreator, architecture))

    def forward(self, x):
        """Does forward propagation"""
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x
