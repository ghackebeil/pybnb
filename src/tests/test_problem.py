from pybnb.common import (minimize,
                          maximize,
                          inf)
from pybnb.problem import Problem

import pytest

class MinProblem(Problem):
    def sense(self): return minimize

class MaxProblem(Problem):
    def sense(self): return maximize

class TestProblem(object):

    def test_infeasible_objective(self):
        p = MinProblem()
        assert p.infeasible_objective() == inf
        p = MaxProblem()
        assert p.infeasible_objective() == -inf

    def test_unbounded_objective(self):
        p = MinProblem()
        assert p.unbounded_objective() == -inf
        p = MaxProblem()
        assert p.unbounded_objective() == inf

    def test_default_methods(self):
        p = Problem()
        with pytest.raises(NotImplementedError):
            p.sense()
        with pytest.raises(NotImplementedError):
            p.objective()
        with pytest.raises(NotImplementedError):
            p.load_state(None)
        with pytest.raises(NotImplementedError):
            p.save_state(None)
        with pytest.raises(NotImplementedError):
            p.branch(None)
        p.notify_new_best_objective_received(None, None)
        p.notify_new_best_objective(None, None)
        p.notify_solve_finished(None, None, None)
