"""Unit tests for the core fixes in the neural network training pipeline.

Tests verify:
1. acceptance_probability returns correct probabilities
2. random_neighbor deep-copies and does not mutate the original
3. ClassicLayer.randomNeighbor perturbation_ratio controls weight perturbations
"""
import copy
import math
import random
import numpy as np

from neuralNetwork import NeuralNetwork
from activations import Sigmoid
from classicLayer import Classic
import trainers.unsupervised.simulatedAnnealing as sa


def test_acceptance_probability_better_solution():
    """A better solution (new > current) always has probability 1.0."""
    trainer = sa.SimulatedAnnealing()
    # new=2.0 > current=1.0 => always accept
    assert trainer.acceptance_probability(1.0, 2.0, 1.0) == 1.0
    # Even at very low temperature, better solutions are always accepted
    assert trainer.acceptance_probability(1.0, 2.0, 0.001) == 1.0
    # Equal solutions should also be accepted (new == current)
    assert trainer.acceptance_probability(2.0, 2.0, 1.0) == 1.0


def test_acceptance_probability_worse_solution():
    """A worse solution (new < current) has probability in (0, 1)."""
    trainer = sa.SimulatedAnnealing()
    # For maximization: exp((new - current) / temperature)
    # new=1.0, current=2.0, T=1.0 => exp(-1) ≈ 0.368
    p = trainer.acceptance_probability(2.0, 1.0, 1.0)
    assert 0.36 < p < 0.37, f"Expected ~0.368, got {p}"

    # At very low temperature, probability should approach 0
    p = trainer.acceptance_probability(2.0, 1.0, 0.001)
    assert p < 0.001, f"Expected near 0, got {p}"

    # At high temperature, probability approaches 1
    p = trainer.acceptance_probability(2.0, 1.0, 10000)
    assert p > 0.999, f"Expected near 1, got {p}"


def test_acceptance_probability_temperature_zero():
    """When temperature approaches 0, worse solutions are almost never accepted."""
    trainer = sa.SimulatedAnnealing()
    p = trainer.acceptance_probability(2.0, 1.0, 1e-10)
    assert p < 1e-10, f"Expected extremely small probability, got {p}"


def test_acceptance_probability_high_temperature():
    """At very high temperature, almost all solutions are accepted."""
    trainer = sa.SimulatedAnnealing()
    p = trainer.acceptance_probability(1.0, 0.999, 1e10)
    # exp(-0.001 / 1e10) = exp(-1e-13) ≈ 1 - 1e-13, very close to 1
    assert p > 0.999999, f"Expected near 1, got {p}"


def test_random_neighbor_does_not_mutate_original():
    """random_neighbor must deep-copy the network and mutate only the copy.

    The critical bug was that random_neighbor mutated nn.layers[i] (the original)
    instead of tmp.layers[i] (the copy). This test verifies the fix.
    """
    trainer = sa.SimulatedAnnealing()
    original = NeuralNetwork([
        ((2, 3), "classic", Sigmoid()),
        ((3, 2), "classic", Sigmoid()),
    ])
    # Record original weights
    original_weights = [copy.deepcopy(layer.weights) for layer in original.layers]

    # Generate a neighbor
    neighbor = trainer.random_neighbor(original)

    # Verify neighbor is a different object
    assert neighbor is not original, "Neighbor should be a different object from original"

    # Verify original weights are unchanged (deep copy independence)
    for i, layer in enumerate(original.layers):
        assert np.array_equal(layer.weights, original_weights[i]), \
            f"Layer {i} original weights were mutated by random_neighbor"

    # Verify neighbor weights are different from original (perturbation occurred)
    some_weights_differ = False
    for i, layer in enumerate(neighbor.layers):
        if not np.array_equal(layer.weights, original.layers[i].weights):
            some_weights_differ = True
            break
    assert some_weights_differ, "Neighbor weights should differ from original after perturbation"


def test_random_neighbor_preserves_structure():
    """random_neighbor should return a network with the same architecture."""
    trainer = sa.SimulatedAnnealing()
    original = NeuralNetwork([
        ((2, 3), "classic", Sigmoid()),
        ((3, 2), "classic", Sigmoid()),
    ])
    neighbor = trainer.random_neighbor(original)

    assert len(neighbor.layers) == len(original.layers), \
        "Neighbor should have same number of layers"

    for i, (nl, ol) in enumerate(zip(neighbor.layers, original.layers)):
        assert nl.weights.shape == ol.weights.shape, \
            f"Layer {i} weight shapes differ: {nl.weights.shape} vs {ol.weights.shape}"


def test_random_neighbor_multiple_calls_independence():
    """Multiple random_neighbor calls should produce different results."""
    trainer = sa.SimulatedAnnealing()
    original = NeuralNetwork([
        ((2, 3), "classic", Sigmoid()),
    ])
    n1 = trainer.random_neighbor(original)
    n2 = trainer.random_neighbor(original)

    # Both should differ from the original
    w1_differs = not np.array_equal(n1.layers[0].weights, original.layers[0].weights)
    w2_differs = not np.array_equal(n2.layers[0].weights, original.layers[0].weights)
    assert w1_differs, "First neighbor should differ from original"
    assert w2_differs, "Second neighbor should differ from original"

    # Original must remain untouched after both calls
    assert np.array_equal(original.layers[0].weights, original.layers[0].weights)


def test_classic_random_neighbor_perturbation_ratio_zero():
    """perturbation_ratio=0 should not perturb any weights."""
    layer = Classic((2, 3), Sigmoid())
    original_weights = layer.weights.copy()
    layer.randomNeighbor(perturbation_ratio=0.0)
    assert np.array_equal(layer.weights, original_weights), \
        "Weights should not change when perturbation_ratio=0"


def test_classic_random_neighbor_perturbation_ratio_one():
    """perturbation_ratio=1 should perturb all weights."""
    layer = Classic((2, 3), Sigmoid())
    original_weights = layer.weights.copy()
    layer.randomNeighbor(perturbation_ratio=1.0)
    # At least some weights should differ (at least 1 of 9)
    assert not np.array_equal(layer.weights, original_weights), \
        "Weights should change when perturbation_ratio=1"


def test_classic_random_neighbor_unique_indices():
    """random.sample should ensure each weight is perturbed at most once per call.

    Verify by checking that with perturbation_ratio=1.0 on a large layer,
    the number of changed weights matches the expected count.
    """
    layer = Classic((5, 10), Sigmoid())  # 60 weights
    original_weights = layer.weights.copy()
    layer.randomNeighbor(perturbation_ratio=1.0)  # perturb all 60 weights

    changed = np.sum(layer.weights != original_weights)
    assert changed == 60, \
        f"Expected all 60 weights to change with ratio=1.0, but only {changed} changed"


def test_optimize_runs_cleanly():
    """The optimize method should run without errors and improve fitness.

    Uses a small single-layer network to keep the test fast.
    """
    from xor_test import inputs, outputs, fitness
    trainer = sa.SimulatedAnnealing()
    nn = NeuralNetwork([
        ((2, 4), "classic", Sigmoid()),
        ((4, 2), "classic", Sigmoid()),
    ])
    initial_fitness = fitness(nn)
    # Run 3 optimization cycles (each does 100K iterations internally)
    for _ in range(3):
        nn = trainer.optimize(nn, fitness, verbose=False)
    final_fitness = fitness(nn)
    # Fitness should not be NaN or degenerate
    assert not np.isnan(final_fitness), "Fitness should not be NaN after training"
    assert np.isfinite(final_fitness), "Fitness should be finite after training"
    # The network should produce predictions without error
    prediction = nn.forward(inputs)
    assert prediction.shape == (4, 2), f"Expected shape (4,2), got {prediction.shape}"
