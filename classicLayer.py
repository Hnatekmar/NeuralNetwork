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

    def randomNeighbor(self, perturbation_ratio=0.1):
        """
        Generate a neighbor state by perturbing a fraction of weights.

        Each call perturbs a subset of weights (controlled by perturbation_ratio)
        by adding a small random delta in [-alpha, alpha]. The bias row (index 0)
        is intentionally left mutable — like all other weights, biases are perturbed
        during neighbor generation and are NOT reset, allowing the optimizer to
        learn bias values rather than keeping them fixed at initialization.

        Args:
            perturbation_ratio: Fraction of weights to perturb per call (default 0.1 = 10%).
        """
        alpha = 0.1
        shape = self.weights.shape
        n_weights = shape[0] * shape[1]
        num_perturbations = int(n_weights * perturbation_ratio)
        if num_perturbations == 0 and perturbation_ratio > 0:
            num_perturbations = 1
        if num_perturbations > 0:
            # Use random.sample to ensure each weight is perturbed at most once per call,
            # preventing duplicate perturbations from reducing exploration diversity.
            indices = random.sample(range(n_weights), min(num_perturbations, n_weights))
            for idx in indices:
                i = idx // shape[1]
                j = idx % shape[1]
                self.weights[i, j] += (random.random() * 2.0 - 1.0) * alpha

