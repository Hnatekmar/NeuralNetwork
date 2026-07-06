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

def test_xor(max_cycles=20, convergence_threshold=3.999):
    trainer = sa.SimulatedAnnealing()
    nn = NeuralNetwork([
        ((2, 4), "classic", Sigmoid()),
        ((4, 2), "classic", Sigmoid())
    ])
    for cycle in range(max_cycles):
        verbose = (cycle % 5 == 0)
        nn = trainer.optimize(nn, fitness, verbose=verbose)
        prediction = nn.forward(inputs)
        result = np.argmax(prediction, axis=1)
        current_fitness = fitness(nn)
        print(f"Cycle {cycle + 1}/{max_cycles} | Fitness: {current_fitness:.4f} | Argmax: {result}")
        if current_fitness >= convergence_threshold:
            print("✓ Converged!")
            print(f"Final predictions:\n{np.round(prediction, 4)}")
            break
    return nn

if __name__ == '__main__':
    trained_nn = test_xor()
