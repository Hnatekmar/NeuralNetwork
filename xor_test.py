"""Tests neural network implementation on xor problem"""
import numpy as np
from neuralNetwork import *
from activations import *
import trainers.unsupervised.simulatedAnnealing as sa
import lossFunctions as lf

inputs = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]])
outputs = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0]])

def fitness(nn):
    prediction = nn.forward(inputs)
    return 4.0 - lf.mse(outputs, prediction)

def test_xor():
    trainer = sa.SimulatedAnnealing()
    nn = NeuralNetwork([
        ((2, 2), "classic", Sigmoid()),
        ((2, 2), "classic", Sigmoid())
    ])
    # TODO: Training
    while True:
        nn = trainer.optimize(nn, fitness)
        prediction = nn.forward(inputs)
        print(prediction)
        result = np.argmax(prediction, axis=1)
        print(result)
        if np.array_equal(result, [0, 1, 1, 0]):
            break

if __name__ == '__main__':
    test_xor()
