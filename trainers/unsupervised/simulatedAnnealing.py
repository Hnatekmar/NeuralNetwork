import copy
import math
import random

class SimulatedAnnealing:
    def random_neighbor(self, nn):
        tmp = copy.deepcopy(nn)
        for i in range(len(tmp.layers)):
            tmp.layers[i].randomNeighbor()
        return tmp

    def acceptance_probability(self, current, new, temperature):
        if new > current:
            return 1.0
        # For maximization: probability of accepting a worse solution decreases
        # as temperature drops. new < current => exponent negative => P in (0, 1).
        return math.exp((new - current) / temperature)

    def optimize(self, nn, fitness, verbose=False, log_interval=10000):
        t = 100000
        cooling = 1.0 - 0.0001
        guess = copy.deepcopy(nn)
        score = fitness(guess)
        bestGuess = copy.deepcopy(nn)
        bestFitness = score
        epsilon = 1
        iteration = 0
        while t > epsilon:
            neighbor = self.random_neighbor(guess)
            neighborScore = fitness(neighbor)
            if self.acceptance_probability(score, neighborScore, t) > random.random():
                guess = neighbor
                score = neighborScore
                if score > bestFitness:
                    bestFitness = score
                    bestGuess = copy.deepcopy(guess)
            t *= cooling
            iteration += 1
            if verbose and iteration % log_interval == 0:
                print(f"  Iteration {iteration}, temperature={t:.4f}, score={score:.4f}, bestFitness={bestFitness:.4f}")
        return bestGuess
