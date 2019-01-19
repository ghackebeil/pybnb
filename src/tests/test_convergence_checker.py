import itertools
import math

from pybnb.common import (minimize,
                          maximize,
                          inf,
                          TerminationCondition)
from pybnb.convergence_checker import ConvergenceChecker

class TestConvergenceChecker(object):

    def test_sense(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.sense == minimize
        # max
        p = ConvergenceChecker(maximize)
        assert p.sense == maximize

    def test_infeasible_objective(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.infeasible_objective == inf
        # max
        p = ConvergenceChecker(maximize)
        assert p.infeasible_objective == -inf

    def test_unbounded_objective(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.unbounded_objective == -inf
        # max
        p = ConvergenceChecker(maximize)
        assert p.unbounded_objective == inf

    def test_check_termination_criteria(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.check_termination_criteria(inf, None) is \
            TerminationCondition.optimality
        assert p.check_termination_criteria(0, 0) is \
            TerminationCondition.optimality
        assert p.check_termination_criteria(0, 1) is None
        p = ConvergenceChecker(minimize,
                               objective_stop=1)
        assert p.check_termination_criteria(0, 1.1) is None
        assert p.check_termination_criteria(0, 1) is \
            TerminationCondition.objective_limit
        p = ConvergenceChecker(minimize,
                               objective_stop=inf)
        assert p.check_termination_criteria(-inf, inf) is None
        assert p.check_termination_criteria(-inf, 1000000) is \
            TerminationCondition.objective_limit
        p = ConvergenceChecker(minimize,
                               bound_stop=0)
        assert p.check_termination_criteria(-0.1, 1) is None
        assert p.check_termination_criteria(0, 1) is \
            TerminationCondition.objective_limit
        p = ConvergenceChecker(minimize,
                               bound_stop=-inf)
        assert p.check_termination_criteria(-inf, inf) is None
        assert p.check_termination_criteria(-10000000, inf) is \
            TerminationCondition.objective_limit
        # max
        p = ConvergenceChecker(maximize)
        assert p.check_termination_criteria(-inf, None) is \
            TerminationCondition.optimality
        assert p.check_termination_criteria(0, 0) is \
            TerminationCondition.optimality
        assert p.check_termination_criteria(0, -1) is None
        p = ConvergenceChecker(maximize,
                               objective_stop=-1)
        assert p.check_termination_criteria(0, -1.1) is None
        assert p.check_termination_criteria(0, -1) is \
            TerminationCondition.objective_limit
        p = ConvergenceChecker(maximize,
                               objective_stop=-inf)
        assert p.check_termination_criteria(inf, -inf) is None
        assert p.check_termination_criteria(inf, -1000000) is \
            TerminationCondition.objective_limit
        p = ConvergenceChecker(maximize,
                               bound_stop=0)
        assert p.check_termination_criteria(0.1, -1) is None
        assert p.check_termination_criteria(0, -1) is \
            TerminationCondition.objective_limit
        p = ConvergenceChecker(maximize,
                               bound_stop=inf)
        assert p.check_termination_criteria(inf, -inf) is None
        assert p.check_termination_criteria(1000000, -inf) is \
            TerminationCondition.objective_limit

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
            p = ConvergenceChecker(sense,
                                   absolute_gap=0,
                                   relative_gap=0)
            assert p.objective_is_optimal(0,0)
            assert p.objective_is_optimal(1,1)
            assert p.objective_is_optimal(-1,-1)

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

    def test_eligible_for_queue(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.eligible_for_queue(-inf, -inf)
        assert p.eligible_for_queue(-inf, inf)
        assert p.eligible_for_queue(-inf, 0.0)
        assert not p.eligible_for_queue(inf, -inf)
        assert not p.eligible_for_queue(inf, inf)
        assert not p.eligible_for_queue(inf, 0.0)
        assert not p.eligible_for_queue(0.0, -inf)
        assert p.eligible_for_queue(0.0, inf)
        assert p.eligible_for_queue(-1e-16, 0.0)
        assert not p.eligible_for_queue(0.0, 0.0)
        p = ConvergenceChecker(minimize,
                               queue_tolerance=0.1)
        assert not p.eligible_for_queue(-0.1, 0.0)
        assert p.eligible_for_queue(-0.11, 0.0)
        p = ConvergenceChecker(minimize,
                               queue_tolerance=None)
        assert p.eligible_for_queue(0.0, 0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.eligible_for_queue(-inf, -inf)
        assert not p.eligible_for_queue(-inf, inf)
        assert not p.eligible_for_queue(-inf, 0.0)
        assert p.eligible_for_queue(inf, -inf)
        assert not p.eligible_for_queue(inf, inf)
        assert p.eligible_for_queue(inf, 0.0)
        assert p.eligible_for_queue(0.0, -inf)
        assert not p.eligible_for_queue(0.0, inf)
        assert p.eligible_for_queue(1e-16, 0.0)
        assert not p.eligible_for_queue(0.0, 0.0)
        p = ConvergenceChecker(maximize,
                               queue_tolerance=0.1)
        assert not p.eligible_for_queue(0.1, 0.0)
        assert p.eligible_for_queue(0.11, 0.0)
        p = ConvergenceChecker(maximize,
                               queue_tolerance=None)
        assert p.eligible_for_queue(0.0, 0.0)

    def test_eligible_to_branch(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.eligible_to_branch(-inf, -inf)
        assert p.eligible_to_branch(-inf, inf)
        assert p.eligible_to_branch(-inf, 0.0)
        assert not p.eligible_to_branch(inf, -inf)
        assert not p.eligible_to_branch(inf, inf)
        assert not p.eligible_to_branch(inf, 0.0)
        assert not p.eligible_to_branch(0.0, -inf)
        assert p.eligible_to_branch(0.0, inf)
        assert not p.eligible_to_branch(0.0, 0.0)
        p = ConvergenceChecker(minimize,
                               branch_tolerance=0.1)
        assert not p.eligible_to_branch(-0.1, 0.0)
        assert p.eligible_to_branch(-0.11, 0.0)
        p = ConvergenceChecker(minimize,
                               branch_tolerance=None)
        assert p.eligible_to_branch(0.0, 0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.eligible_to_branch(-inf, -inf)
        assert not p.eligible_to_branch(-inf, inf)
        assert not p.eligible_to_branch(-inf, 0.0)
        assert p.eligible_to_branch(inf, -inf)
        assert not p.eligible_to_branch(inf, inf)
        assert p.eligible_to_branch(inf, 0.0)
        assert p.eligible_to_branch(0.0, -inf)
        assert not p.eligible_to_branch(0.0, inf)
        assert not p.eligible_to_branch(0.0, 0.0)
        p = ConvergenceChecker(maximize,
                               branch_tolerance=0.1)
        assert not p.eligible_to_branch(0.1, 0.0)
        assert p.eligible_to_branch(0.11, 0.0)
        p = ConvergenceChecker(maximize,
                               branch_tolerance=None)
        assert p.eligible_to_branch(0.0, 0.0)

    def test_bound_worsened(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.bound_worsened(-inf, -inf)
        assert p.bound_worsened(-inf, inf)
        assert p.bound_worsened(-inf, 0.0)
        assert not p.bound_worsened(inf, -inf)
        assert not p.bound_worsened(inf, inf)
        assert not p.bound_worsened(inf, 0.0)
        assert not p.bound_worsened(0.0, -inf)
        assert p.bound_worsened(0.0, inf)
        assert not p.bound_worsened(0.0, 0.0)
        p = ConvergenceChecker(minimize,
                               comparison_tolerance=0.1)
        assert not p.bound_worsened(-0.1, 0.0)
        assert p.bound_worsened(-0.11, 0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.bound_worsened(-inf, -inf)
        assert not p.bound_worsened(-inf, inf)
        assert not p.bound_worsened(-inf, 0.0)
        assert p.bound_worsened(inf, -inf)
        assert not p.bound_worsened(inf, inf)
        assert p.bound_worsened(inf, 0.0)
        assert p.bound_worsened(0.0, -inf)
        assert not p.bound_worsened(0.0, inf)
        assert not p.bound_worsened(0.0, 0.0)
        p = ConvergenceChecker(maximize,
                               comparison_tolerance=0.1)
        assert not p.bound_worsened(0.1, 0.0)
        assert p.bound_worsened(0.11, 0.0)

    def test_objective_improved(self):
        # min
        p = ConvergenceChecker(minimize)
        assert not p.objective_improved(-inf, -inf)
        assert p.objective_improved(-inf, inf)
        assert p.objective_improved(-inf, 0.0)
        assert not p.objective_improved(inf, -inf)
        assert not p.objective_improved(inf, inf)
        assert not p.objective_improved(inf, 0.0)
        assert not p.objective_improved(0.0, -inf)
        assert p.objective_improved(0.0, inf)
        assert not p.objective_improved(0.0, 0.0)
        p = ConvergenceChecker(minimize,
                               comparison_tolerance=0.1)
        assert not p.objective_improved(-0.1, 0.0)
        assert p.objective_improved(-0.11, 0.0)
        # max
        p = ConvergenceChecker(maximize)
        assert not p.objective_improved(-inf, -inf)
        assert not p.objective_improved(-inf, inf)
        assert not p.objective_improved(-inf, 0.0)
        assert p.objective_improved(inf, -inf)
        assert not p.objective_improved(inf, inf)
        assert p.objective_improved(inf, 0.0)
        assert p.objective_improved(0.0, -inf)
        assert not p.objective_improved(0.0, inf)
        assert not p.objective_improved(0.0, 0.0)
        p = ConvergenceChecker(maximize,
                               comparison_tolerance=0.1)
        assert not p.objective_improved(0.1, 0.0)
        assert p.objective_improved(0.11, 0.0)

    def test_worst_bound(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.worst_bound(-1, 0, 1) == -1
        assert p.worst_bound([-1, 0, 1]) == -1
        # max
        p = ConvergenceChecker(maximize)
        assert p.worst_bound(-1, 0, 1) == 1
        assert p.worst_bound([-1, 0, 1]) == 1

    def test_best_bound(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.best_bound(-1, 0, 1) == 1
        assert p.best_bound([-1, 0, 1]) == 1
        # max
        p = ConvergenceChecker(maximize)
        assert p.best_bound(-1, 0, 1) == -1
        assert p.best_bound([-1, 0, 1]) == -1

    def test_worst_objective(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.worst_objective(-1, 0, 1) == 1
        assert p.worst_objective([-1, 0, 1]) == 1
        # max
        p = ConvergenceChecker(maximize)
        assert p.worst_objective(-1, 0, 1) == -1
        assert p.worst_objective([-1, 0, 1]) == -1

    def test_best_objective(self):
        # min
        p = ConvergenceChecker(minimize)
        assert p.best_objective(-1, 0, 1) == -1
        assert p.best_objective([-1, 0, 1]) == -1
        # max
        p = ConvergenceChecker(maximize)
        assert p.best_objective(-1, 0, 1) == 1
        assert p.best_objective([-1, 0, 1]) == 1
