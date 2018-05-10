import itertools
import math

from pybnb.common import (minimize,
                          maximize,
                          inf)
from pybnb.convergence_checker import ConvergenceChecker

class TestConvergenceChecker(object):

    def test_sense(self):
        p = ConvergenceChecker(minimize)
        assert p.sense == minimize
        p = ConvergenceChecker(maximize)
        assert p.sense == maximize

    def test_infeasible_objective(self):
        p = ConvergenceChecker(minimize)
        assert p.infeasible_objective == inf
        p = ConvergenceChecker(maximize)
        assert p.infeasible_objective == -inf

    def test_unbounded_objective(self):
        p = ConvergenceChecker(minimize)
        assert p.unbounded_objective == -inf
        p = ConvergenceChecker(maximize)
        assert p.unbounded_objective == inf

    def test_objective_is_optimal(self):

        for sense in [minimize, maximize]:
            p = ConvergenceChecker(sense)
            for bound,objective in itertools.product([-inf,
                                                      inf,
                                                      0.0],
                                                     [-inf,
                                                      inf,
                                                      0.0]):
                if math.isinf(bound) and math.isinf(objective):
                    if bound != p.infeasible_objective:
                        assert not p.objective_is_optimal(objective, bound)
                elif objective == bound:
                    assert p.objective_is_optimal(objective, bound)
                elif bound != p.infeasible_objective:
                    assert not p.objective_is_optimal(objective, bound)

    def test_compute_relative_gap(self):

        for sense in [minimize, maximize]:
            p = ConvergenceChecker(sense)
            for bound,objective in itertools.product([-inf,
                                                      inf,
                                                      0.0],
                                                     [-inf,
                                                      inf,
                                                      0.0]):
                if (not math.isinf(bound)) or \
                   (not math.isinf(objective)):
                    continue
                if bound == objective:
                    assert p.compute_relative_gap(bound,objective) == 0
                elif p.sense == minimize:
                    if (bound == -inf) or \
                       (objective == inf):
                        assert p.compute_relative_gap(bound,objective) == \
                            inf
                    else:
                        assert p.compute_relative_gap(bound,objective) == \
                            -inf
                else:
                    assert p.sense == maximize
                    if (bound == inf) or \
                       (objective == -inf):
                        assert p.compute_relative_gap(bound,objective) == \
                            inf
                    else:
                        assert p.compute_relative_gap(bound,objective) == \
                            -inf

    def test_bound_improved(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.bound_improved(-inf,
                                    -inf)
        assert p.bound_improved(-inf,
                                inf)
        assert p.bound_improved(-inf,
                                0.0)
        assert not p.bound_improved(inf,
                                    -inf)
        assert not p.bound_improved(inf,
                                    inf)
        assert not p.bound_improved(inf,
                                    0.0)
        assert not p.bound_improved(0.0,
                                    -inf)
        assert p.bound_improved(0.0,
                                inf)
        assert not p.bound_improved(0.0, 0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.bound_improved(-inf,
                                    -inf)
        assert not p.bound_improved(-inf,
                                    inf)
        assert not p.bound_improved(-inf,
                                    0.0)
        assert p.bound_improved(inf,
                                -inf)
        assert not p.bound_improved(inf,
                                    inf)
        assert p.bound_improved(inf,
                                0.0)
        assert p.bound_improved(0.0,
                                -inf)
        assert not p.bound_improved(0.0,
                                    inf)
        assert not p.bound_improved(0.0,
                                    0.0)

    def test_bound_worsened(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.bound_worsened(-inf,
                                -inf)
        assert p.bound_worsened(-inf,
                                inf)
        assert p.bound_worsened(-inf,
                                0.0)
        assert not p.bound_worsened(inf,
                                    -inf)
        assert not p.bound_worsened(inf,
                                    inf)
        assert not p.bound_worsened(inf,
                                    0.0)
        assert not p.bound_worsened(0.0,
                                    -inf)
        assert p.bound_worsened(0.0,
                                inf)
        assert not p.bound_worsened(0.0,
                                    0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.bound_worsened(-inf,
                                    -inf)
        assert not p.bound_worsened(-inf,
                                    inf)
        assert not p.bound_worsened(-inf,
                                    0.0)
        assert p.bound_worsened(inf,
                                -inf)
        assert not p.bound_worsened(inf,
                                    inf)
        assert p.bound_worsened(inf,
                                0.0)
        assert p.bound_worsened(0.0,
                                -inf)
        assert not p.bound_worsened(0.0,
                                    inf)
        assert not p.bound_worsened(0.0,
                                    0.0)

    def test_objective_improved(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.objective_improved(-inf,
                                        -inf)
        assert p.objective_improved(-inf,
                                    inf)
        assert p.objective_improved(-inf,
                                    0.0)
        assert not p.objective_improved(inf,
                                        -inf)
        assert not p.objective_improved(inf,
                                        inf)
        assert not p.objective_improved(inf,
                                        0.0)
        assert not p.objective_improved(0.0,
                                        -inf)
        assert p.objective_improved(0.0,
                                    inf)
        assert not p.objective_improved(0.0,
                                        0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.objective_improved(-inf,
                                        -inf)
        assert not p.objective_improved(-inf,
                                        inf)
        assert not p.objective_improved(-inf,
                                        0.0)
        assert p.objective_improved(inf,
                                    -inf)
        assert not p.objective_improved(inf,
                                        inf)
        assert p.objective_improved(inf,
                                    0.0)
        assert p.objective_improved(0.0,
                                    -inf)
        assert not p.objective_improved(0.0,
                                        inf)
        assert not p.objective_improved(0.0,
                                        0.0)

    def test_objective_can_improve(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.objective_can_improve(-inf,
                                           -inf)
        assert not p.objective_can_improve(-inf,
                                           inf)
        assert not p.objective_can_improve(-inf,
                                           0.0)
        assert p.objective_can_improve(inf,
                                       -inf)
        assert not p.objective_can_improve(inf,
                                           inf)
        assert p.objective_can_improve(inf,
                                       0.0)
        assert p.objective_can_improve(0.0,
                                       -inf)
        assert not p.objective_can_improve(0.0,
                                           inf)
        assert not p.objective_can_improve(0.0,
                                           0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.objective_can_improve(-inf,
                                           -inf)
        assert p.objective_can_improve(-inf,
                                       inf)
        assert p.objective_can_improve(-inf,
                                       0.0)
        assert not p.objective_can_improve(inf,
                                           -inf)
        assert not p.objective_can_improve(inf,
                                           inf)
        assert not p.objective_can_improve(inf,
                                           0.0)
        assert not p.objective_can_improve(0.0,
                                           -inf)
        assert p.objective_can_improve(0.0,
                                       inf)
        assert not p.objective_can_improve(0.0,
                                           0.0)

    def test_bound_is_suboptimal(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.bound_is_suboptimal(-inf,
                                         -inf)
        assert not p.bound_is_suboptimal(-inf,
                                         inf)
        assert not p.bound_is_suboptimal(-inf,
                                         0.0)
        assert p.bound_is_suboptimal(inf,
                                     -inf)
        assert not p.bound_is_suboptimal(inf,
                                         inf)
        assert p.bound_is_suboptimal(inf,
                                     0.0)
        assert p.bound_is_suboptimal(0.0,
                                     -inf)
        assert not p.bound_is_suboptimal(0.0,
                                         inf)
        assert not p.bound_is_suboptimal(0.0,
                                         0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.bound_is_suboptimal(-inf,
                                         -inf)
        assert p.bound_is_suboptimal(-inf,
                                     inf)
        assert p.bound_is_suboptimal(-inf,
                                     0.0)
        assert not p.bound_is_suboptimal(inf,
                                         -inf)
        assert not p.bound_is_suboptimal(inf,
                                         inf)
        assert not p.bound_is_suboptimal(inf,
                                         0.0)
        assert not p.bound_is_suboptimal(0.0,
                                         -inf)
        assert p.bound_is_suboptimal(0.0,
                                     inf)
        assert not p.bound_is_suboptimal(0.0,
                                         0.0)
