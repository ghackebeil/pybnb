import sys
import os
import random
import math

thisdir = os.path.dirname(os.path.abspath(__file__))
exdir = os.path.join(
    os.path.dirname(os.path.dirname(thisdir)),
    "examples")
tspdir = os.path.join(exdir,"scripts","tsp")

sys.path.insert(0,tspdir)
try:
    from tsp_util import run_2opt
finally:
    sys.path.remove(tspdir)

class TestHeuristics(object):

    def _generate_random_matrix(self, size, symmetric=False):
        assert size > 1
        dist = [[None for j in range(size)]
               for i in range(size)]
        for i in range(size):
            dist[i][i] = 0
            for j in range(i):
                assert i != j
                dist[i][j] = random.randint(1,1000)
                if symmetric:
                    dist[j][i] = dist[i][j]
                else:
                    dist[j][i] = random.randint(1,1000)
        return dist

    def _compute_route_cost(self, dist, route):
        N = len(route)
        assert N == len(dist)+1
        assert route[0] == route[-1]
        cost = 0
        for k in range(N-1):
            assert route[k] != route[k+1]
            assert dist[route[k]][route[k+1]] != 0
            cost += dist[route[k]][route[k+1]]
        assert not math.isinf(cost)
        assert not math.isnan(cost)
        return cost

    def test_run_2opt(self):
        for n in range(1,7):
            size = 2**n
            for dist in (self._generate_random_matrix(size, symmetric=True),
                         self._generate_random_matrix(size, symmetric=False)):
                route = list(range(size))
                route.append(route[0])
                cost = self._compute_route_cost(dist, route)
                new_cost, new_route = run_2opt(dist, cost, route)
                if new_route != route:
                    assert new_cost < cost
                new_cost_ = self._compute_route_cost(dist, new_route)
                assert new_cost_ == new_cost
