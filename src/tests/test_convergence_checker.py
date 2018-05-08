import itertools

from pybnb.common import (minimize,
                          maximize,
                          infinity,
                          is_infinite)
from pybnb.convergence_checker import ConvergenceChecker

class TestConvergenceChecker(object):

    def test_sense(self):
        p = ConvergenceChecker(minimize)
        assert p.sense == minimize
        p = ConvergenceChecker(maximize)
        assert p.sense == maximize

    def test_infeasible_objective(self):
        p = ConvergenceChecker(minimize)
        assert p.infeasible_objective == infinity
        p = ConvergenceChecker(maximize)
        assert p.infeasible_objective == -infinity

    def test_unbounded_objective(self):
        p = ConvergenceChecker(minimize)
        assert p.unbounded_objective == -infinity
        p = ConvergenceChecker(maximize)
        assert p.unbounded_objective == infinity

    def test_objective_is_optimal(self):

        for sense in [minimize, maximize]:
            p = ConvergenceChecker(sense)
            for bound,objective in itertools.product([-infinity,
                                                      infinity,
                                                      0.0],
                                                     [-infinity,
                                                      infinity,
                                                      0.0]):
                if is_infinite(bound) and is_infinite(objective):
                    if bound != p.infeasible_objective:
                        assert not p.objective_is_optimal(objective, bound)
                elif objective == bound:
                    assert p.objective_is_optimal(objective, bound)
                elif bound != p.infeasible_objective:
                    assert not p.objective_is_optimal(objective, bound)

    def test_compute_relative_gap(self):

        for sense in [minimize, maximize]:
            p = ConvergenceChecker(sense)
            for bound,objective in itertools.product([-infinity,
                                                      infinity,
                                                      0.0],
                                                     [-infinity,
                                                      infinity,
                                                      0.0]):
                if (not is_infinite(bound)) or \
                   (not is_infinite(objective)):
                    continue
                if bound == objective:
                    assert p.compute_relative_gap(bound,objective) == 0
                elif p.sense == minimize:
                    if (bound == -infinity) or \
                       (objective == infinity):
                        assert p.compute_relative_gap(bound,objective) == \
                            infinity
                    else:
                        assert p.compute_relative_gap(bound,objective) == \
                            -infinity
                else:
                    assert p.sense == maximize
                    if (bound == infinity) or \
                       (objective == -infinity):
                        assert p.compute_relative_gap(bound,objective) == \
                            infinity
                    else:
                        assert p.compute_relative_gap(bound,objective) == \
                            -infinity

    def test_bound_improved(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.bound_improved(-infinity,
                                    -infinity)
        assert p.bound_improved(-infinity,
                                infinity)
        assert p.bound_improved(-infinity,
                                0.0)
        assert not p.bound_improved(infinity,
                                    -infinity)
        assert not p.bound_improved(infinity,
                                    infinity)
        assert not p.bound_improved(infinity,
                                    0.0)
        assert not p.bound_improved(0.0,
                                    -infinity)
        assert p.bound_improved(0.0,
                                infinity)
        assert not p.bound_improved(0.0, 0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.bound_improved(-infinity,
                                    -infinity)
        assert not p.bound_improved(-infinity,
                                    infinity)
        assert not p.bound_improved(-infinity,
                                    0.0)
        assert p.bound_improved(infinity,
                                -infinity)
        assert not p.bound_improved(infinity,
                                    infinity)
        assert p.bound_improved(infinity,
                                0.0)
        assert p.bound_improved(0.0,
                                -infinity)
        assert not p.bound_improved(0.0,
                                    infinity)
        assert not p.bound_improved(0.0,
                                    0.0)

    def test_bound_worsened(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.bound_worsened(-infinity,
                                -infinity)
        assert p.bound_worsened(-infinity,
                                infinity)
        assert p.bound_worsened(-infinity,
                                0.0)
        assert not p.bound_worsened(infinity,
                                    -infinity)
        assert not p.bound_worsened(infinity,
                                    infinity)
        assert not p.bound_worsened(infinity,
                                    0.0)
        assert not p.bound_worsened(0.0,
                                    -infinity)
        assert p.bound_worsened(0.0,
                                infinity)
        assert not p.bound_worsened(0.0,
                                    0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.bound_worsened(-infinity,
                                    -infinity)
        assert not p.bound_worsened(-infinity,
                                    infinity)
        assert not p.bound_worsened(-infinity,
                                    0.0)
        assert p.bound_worsened(infinity,
                                -infinity)
        assert not p.bound_worsened(infinity,
                                    infinity)
        assert p.bound_worsened(infinity,
                                0.0)
        assert p.bound_worsened(0.0,
                                -infinity)
        assert not p.bound_worsened(0.0,
                                    infinity)
        assert not p.bound_worsened(0.0,
                                    0.0)

    def test_objective_improved(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.objective_improved(-infinity,
                                        -infinity)
        assert p.objective_improved(-infinity,
                                    infinity)
        assert p.objective_improved(-infinity,
                                    0.0)
        assert not p.objective_improved(infinity,
                                        -infinity)
        assert not p.objective_improved(infinity,
                                        infinity)
        assert not p.objective_improved(infinity,
                                        0.0)
        assert not p.objective_improved(0.0,
                                        -infinity)
        assert p.objective_improved(0.0,
                                    infinity)
        assert not p.objective_improved(0.0,
                                        0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.objective_improved(-infinity,
                                        -infinity)
        assert not p.objective_improved(-infinity,
                                        infinity)
        assert not p.objective_improved(-infinity,
                                        0.0)
        assert p.objective_improved(infinity,
                                    -infinity)
        assert not p.objective_improved(infinity,
                                        infinity)
        assert p.objective_improved(infinity,
                                    0.0)
        assert p.objective_improved(0.0,
                                    -infinity)
        assert not p.objective_improved(0.0,
                                        infinity)
        assert not p.objective_improved(0.0,
                                        0.0)

    def test_objective_can_improve(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.objective_can_improve(-infinity,
                                           -infinity)
        assert not p.objective_can_improve(-infinity,
                                           infinity)
        assert not p.objective_can_improve(-infinity,
                                           0.0)
        assert p.objective_can_improve(infinity,
                                       -infinity)
        assert not p.objective_can_improve(infinity,
                                           infinity)
        assert p.objective_can_improve(infinity,
                                       0.0)
        assert p.objective_can_improve(0.0,
                                       -infinity)
        assert not p.objective_can_improve(0.0,
                                           infinity)
        assert not p.objective_can_improve(0.0,
                                           0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.objective_can_improve(-infinity,
                                           -infinity)
        assert p.objective_can_improve(-infinity,
                                       infinity)
        assert p.objective_can_improve(-infinity,
                                       0.0)
        assert not p.objective_can_improve(infinity,
                                           -infinity)
        assert not p.objective_can_improve(infinity,
                                           infinity)
        assert not p.objective_can_improve(infinity,
                                           0.0)
        assert not p.objective_can_improve(0.0,
                                           -infinity)
        assert p.objective_can_improve(0.0,
                                       infinity)
        assert not p.objective_can_improve(0.0,
                                           0.0)

    def test_bound_is_suboptimal(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.bound_is_suboptimal(-infinity,
                                         -infinity)
        assert not p.bound_is_suboptimal(-infinity,
                                         infinity)
        assert not p.bound_is_suboptimal(-infinity,
                                         0.0)
        assert p.bound_is_suboptimal(infinity,
                                     -infinity)
        assert not p.bound_is_suboptimal(infinity,
                                         infinity)
        assert p.bound_is_suboptimal(infinity,
                                     0.0)
        assert p.bound_is_suboptimal(0.0,
                                     -infinity)
        assert not p.bound_is_suboptimal(0.0,
                                         infinity)
        assert not p.bound_is_suboptimal(0.0,
                                         0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.bound_is_suboptimal(-infinity,
                                         -infinity)
        assert p.bound_is_suboptimal(-infinity,
                                     infinity)
        assert p.bound_is_suboptimal(-infinity,
                                     0.0)
        assert not p.bound_is_suboptimal(infinity,
                                         -infinity)
        assert not p.bound_is_suboptimal(infinity,
                                         infinity)
        assert not p.bound_is_suboptimal(infinity,
                                         0.0)
        assert not p.bound_is_suboptimal(0.0,
                                         -infinity)
        assert p.bound_is_suboptimal(0.0,
                                     infinity)
        assert not p.bound_is_suboptimal(0.0,
                                         0.0)
