from pybnb.common import (minimize,
                          maximize,
                          infinity)
from pybnb.problem import Problem

class MinProblem(Problem):
    def sense(self): return minimize

class MaxProblem(Problem):
    def sense(self): return maximize

class TestProblem(object):

    def test_infeasible_objective(self):
        p = MinProblem()
        assert p.infeasible_objective == infinity
        p = MaxProblem()
        assert p.infeasible_objective == -infinity

    def test_unbounded_objective(self):
        p = MinProblem()
        assert p.unbounded_objective == -infinity
        p = MaxProblem()
        assert p.unbounded_objective == infinity
