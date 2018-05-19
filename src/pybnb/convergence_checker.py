"""
Convergence checking implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import math

from pybnb.common import (minimize,
                          maximize,
                          inf)

class ConvergenceChecker(object):
    """A class used to check convergence.

    Parameters
    ----------
    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    absolute_gap : float, optional
        The absolute difference between the objective and
        bound that determines optimality. (default: 1e-8)
    relative_gap : float, optional
        The relative difference between the objective and
        bound that determines optimality. (default: 1e-4)
    absolute_tolerance : float, optional
        The absolute tolerance use when deciding if two
        objective or bound values are sufficiently
        different. (default: 1e-10)
    cutoff : float, optional
        If provided, when the best objective is proven worse
        than this value (by greater than
        `absolute_tolerance`), then the cutoff termination
        criteria is met. (default: None)
    """
    __slots__ = ("sense",
                 "absolute_gap_tolerance",
                 "relative_gap_tolerance",
                 "absolute_tolerance",
                 "cutoff",
                 "infeasible_objective",
                 "unbounded_objective")

    def __init__(self,
                 sense,
                 absolute_gap=1e-8,
                 relative_gap=1e-4,
                 absolute_tolerance=1e-10,
                 cutoff=None):
        self.sense = sense
        self.absolute_gap_tolerance = float(absolute_gap)
        self.relative_gap_tolerance = float(relative_gap)
        self.absolute_tolerance = float(absolute_tolerance)
        self.cutoff = None
        if cutoff is not None:
            self.cutoff = float(cutoff)
            assert not math.isinf(self.cutoff)

        if self.sense == minimize:
            self.infeasible_objective = inf
            self.unbounded_objective = -inf
        else:
            assert self.sense == maximize
            self.infeasible_objective = -inf
            self.unbounded_objective = inf
        assert (self.absolute_gap_tolerance >= 0) and \
            (not math.isinf(self.absolute_gap_tolerance))
        assert self.relative_gap_tolerance >= 0 and \
            (not math.isinf(self.relative_gap_tolerance))
        assert self.absolute_tolerance > 0 and \
            (not math.isinf(self.absolute_tolerance))

    def compute_absolute_gap(self, bound, objective):
        """Returns the absolute gap between the bound and
        the objective, respecting the sign relative to the
        objective sense of this problem."""
        if bound == objective:
            return 0.0
        elif math.isinf(bound) or math.isinf(objective):
            if self.sense == minimize:
                if (bound == -inf) or \
                   (objective == inf):
                    return inf
                else:
                    return -inf
            else:
                if (bound == inf) or \
                   (objective == -inf):
                    return inf
                else:
                    return -inf
        else:
            if self.sense == minimize:
                gap = objective - bound
            else:
                gap = bound - objective
            return gap

    def compute_relative_gap(self, bound, objective):
        """Returns the relative gap between the bound and
        the objective, respecting the sign relative to the
        objective sense of this problem."""
        rgap = self.compute_absolute_gap(bound, objective)
        if math.isinf(rgap):
            return rgap
        rgap /= max(1.0, abs(objective))
        return rgap

    def objective_is_optimal(self, objective, bound):
        """Determines if the objective is optimal by
        checking if the optimality gap is below the absolute
        or relative gap tolerances."""
        assert bound != self.infeasible_objective
        if (objective != self.unbounded_objective) and \
           (objective != self.infeasible_objective):
            agap = self.compute_absolute_gap(bound,
                                             objective)
            if agap < self.absolute_gap_tolerance:
                return True
            else:
                rgap = self.compute_relative_gap(bound,
                                                 objective)
                if rgap < self.relative_gap_tolerance:
                    return True
        return False

    def bound_improved(self, new, old):
        """Returns True when the new bound is better than
        the old bound by greater than the absolute
        tolerance."""
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return old > new + self.absolute_tolerance
        else:
            return old < new - self.absolute_tolerance

    def bound_worsened(self, new, old):
        """Returns True when the new bound is worse than the
        old bound by greater than the absolute tolerance."""
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return new < old - self.absolute_tolerance
        else:
            return new > old + self.absolute_tolerance

    def objective_improved(self, new, old):
        """Returns True when the new objective is better
        than the old objective by greater than the absolute
        tolerance."""
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return new < old - self.absolute_tolerance
        else:
            return new > old + self.absolute_tolerance

    def objective_can_improve(self, objective, bound):
        """Returns True when the absolute difference between
        the objective and the bound is greater than the
        absolute tolerance."""
        # handles the both equal and infinite case
        if bound == objective:
            return False
        if self.sense == minimize:
            return bound < objective - self.absolute_tolerance
        else:
            return bound > objective + self.absolute_tolerance

    def bound_is_suboptimal(self, bound, objective):
        """Returns True when bound is worse than the
        objective by greater than the absolute tolerance."""
        # handles the both equal and infinite case
        if bound == objective:
            return False
        if self.sense == minimize:
            return bound > objective + self.absolute_tolerance
        else:
            return bound < objective - self.absolute_tolerance

    def cutoff_is_met(self, bound):
        """Returns true when the bound is better than the
        cutoff value by greater than the absolute
        tolerance. If no cutoff value was provided, this
        method always returns False."""
        if self.cutoff is not None:
            return self.bound_is_suboptimal(bound, self.cutoff)
        return False
