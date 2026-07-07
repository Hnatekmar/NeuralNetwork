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

def test_xor(max_cycles=200, convergence_threshold=3.999, patience=20):
    """Train a neural network on XOR using simulated annealing.

    Args:
        max_cycles: Maximum number of optimization cycles to run.
        convergence_threshold: Fitness value considered "converged".
        patience: Stop early if bestFitness hasn't improved for this many cycles.
    """
    trainer = sa.SimulatedAnnealing()
    nn = NeuralNetwork([
        ((2, 4), "classic", Sigmoid()),
        ((4, 2), "classic", Sigmoid())
    ])
    best_fitness_so_far = -float('inf')
    cycles_without_improvement = 0
    for cycle in range(max_cycles):
        verbose = (cycle % 5 == 0)
        nn = trainer.optimize(nn, fitness, verbose=verbose)
        prediction = nn.forward(inputs)
        result = np.argmax(prediction, axis=1)
        current_fitness = fitness(nn)
        print(f"Cycle {cycle + 1}/{max_cycles} | Fitness: {current_fitness:.4f} | Argmax: {result}")
        if current_fitness > best_fitness_so_far:
            best_fitness_so_far = current_fitness
            cycles_without_improvement = 0
        else:
            cycles_without_improvement += 1
        if current_fitness >= convergence_threshold:
            print("✓ Converged!")
            print(f"Final predictions:\n{np.round(prediction, 4)}")
            break
        if cycles_without_improvement >= patience:
            print(f"⏹ Patience ({patience} cycles) reached — no improvement since cycle {cycle + 1 - patience}")
            break
    return nn

if __name__ == '__main__':
    trained_nn = test_xor()
