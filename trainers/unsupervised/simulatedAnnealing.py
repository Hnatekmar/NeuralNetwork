import copy
import math
import random
import plotly
from plotly.graph_objs import *

class SimulatedAnnealing:
    def random_neighbor(self, nn):
        tmp = copy.deepcopy(nn)
        for i in range(len(nn.layers)):
            nn.layers[i].randomNeighbor()
        return nn

    def acceptance_probability(self, current, new, temperature):
        if new > current:
            return 1.0
        return math.exp((current - new) / temperature)

    def optimize(self, nn, fitness):
        data = []
        t = 100000
        cooling = 1.0 - 0.0001
        guess = nn
        score = fitness(guess)
        bestGuess = nn
        bestFitness = score
        epsilon = 1
        while t > epsilon:
            data.append(score)
            neighbor = self.random_neighbor(nn)
            neighborScore = fitness(neighbor)
            if self.acceptance_probability(score, neighborScore, t) > random.random():
                guess = neighbor
                score = neighborScore
                if score > bestFitness:
                    bestFitness = score
                    bestGuess = guess
            t *= cooling
        plotly.offline.plot({
                "data": [Scatter(y = data)],
                "layout": Layout(title = "fitness")
            })
        return bestGuess
