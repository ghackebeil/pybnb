"""
Convergence checking implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
from __future__ import division

import math

from pybnb.common import (minimize,
                          maximize,
                          inf,
                          TerminationCondition)

def compute_absolute_gap(sense, bound, objective):
    """Returns the absolute gap between the bound and
    the objective, respecting the sign relative to the
    objective sense of this problem."""
    if bound == objective:
        return 0.0
    elif math.isinf(bound) or math.isinf(objective):
        if sense == minimize:
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
        if sense == minimize:
            gap = objective - bound
        else:
            gap = bound - objective
        return gap

def scale_absolute_gap(gap, objective):
    """Convert an absolute gap to a relative gap by scaling
    it by the value `max{1.0,|objective|}`."""
    if math.isinf(gap):
        return gap
    # avoid using abs() as it is slow, for some reason
    if objective > 1.0:
        return gap / objective
    elif objective < -1.0:
        return gap / -objective
    else:
        return gap

def compute_relative_gap(sense, bound, objective):
    """Returns the relative gap between the bound and
    the objective, respecting the sign relative to the
    objective sense of this problem."""
    return scale_absolute_gap(compute_absolute_gap(sense,
                                                   bound,
                                                   objective),
                              objective)

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
    objective_stop : float, optional
        If provided, the "objective_limit" termination
        criteria is met when a feasible objective is found
        that is at least as good as the specified
        value. (default: None)
    bound_stop : float, optional
        If provided, the "objective_limit" termination
        criteria is met when the best bound on the objective
        is at least as good as the specified
        value. (default: None)
    """
    __slots__ = ("sense",
                 "absolute_gap_tolerance",
                 "relative_gap_tolerance",
                 "absolute_tolerance",
                 "objective_stop",
                 "bound_stop",
                 "infeasible_objective",
                 "unbounded_objective")

    def __init__(self,
                 sense,
                 absolute_gap=1e-8,
                 relative_gap=1e-4,
                 absolute_tolerance=1e-10,
                 objective_stop=None,
                 bound_stop=None):
        self.sense = sense
        self.absolute_gap_tolerance = float(absolute_gap)
        self.relative_gap_tolerance = float(relative_gap)
        self.absolute_tolerance = float(absolute_tolerance)
        self.objective_stop = None
        if objective_stop is not None:
            self.objective_stop = float(objective_stop)
            assert not math.isinf(self.objective_stop)
            assert not math.isnan(self.objective_stop)
        self.bound_stop = None
        if bound_stop is not None:
            self.bound_stop = float(bound_stop)
            assert not math.isinf(self.bound_stop)
            assert not math.isnan(self.bound_stop)

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

    def check_termination_criteria(self,
                                   global_bound,
                                   best_objective):
        """Checks if any termination criteria are met and returns
        the corresponding :class:`TerminationCondition
        <pybnb.common.TerminationCondition>` enum value;
        otherwise, `None` is returned."""
        result = None
        if (global_bound == self.infeasible_objective) or \
           (self.objective_is_optimal(best_objective, global_bound)):
            result = TerminationCondition.optimality
        elif self.objective_stop is not None:
            if self.sense == minimize:
                if best_objective <= self.objective_stop:
                    result = TerminationCondition.objective_limit
            else:
                if best_objective >= self.objective_stop:
                    result = TerminationCondition.objective_limit
        elif self.bound_stop is not None:
            if self.sense == minimize:
                if global_bound >= self.bound_stop:
                    result = TerminationCondition.objective_limit
            else:
                if global_bound <= self.bound_stop:
                    result = TerminationCondition.objective_limit
        return result

    def compute_absolute_gap(self, bound, objective):
        """Returns the absolute gap between the bound and
        the objective, respecting the sign relative to the
        objective sense of this problem."""
        return compute_absolute_gap(self.sense, bound, objective)

    def compute_relative_gap(self, bound, objective):
        """Returns the relative gap between the bound and
        the objective, respecting the sign relative to the
        objective sense of this problem."""
        return compute_relative_gap(self.sense, bound, objective)

    def objective_is_optimal(self, objective, bound):
        """Determines if the objective is optimal by
        checking if the optimality gap is below the absolute
        or relative gap tolerances."""
        assert bound != self.infeasible_objective
        if (objective != self.unbounded_objective) and \
           (objective != self.infeasible_objective):
            gap = self.compute_absolute_gap(bound,
                                            objective)
            if gap <= self.absolute_gap_tolerance:
                return True
            else:
                gap = scale_absolute_gap(gap, objective)
                if gap <= self.relative_gap_tolerance:
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
            return new - old > self.absolute_tolerance
        else:
            return old - new > self.absolute_tolerance

    def bound_worsened(self, new, old):
        """Returns True when the new bound is worse than the
        old bound by greater than the absolute tolerance."""
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return old - new > self.absolute_tolerance
        else:
            return new - old > self.absolute_tolerance

    def objective_improved(self, new, old):
        """Returns True when the new objective is better
        than the old objective by greater than the absolute
        tolerance."""
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return old - new > self.absolute_tolerance
        else:
            return new - old > self.absolute_tolerance

    def objective_can_improve(self, objective, bound):
        """Returns True when the absolute difference between
        the objective and the bound is greater than the
        absolute tolerance."""
        # handles the both equal and infinite case
        if bound == objective:
            return False
        if self.sense == minimize:
            return objective - bound > self.absolute_tolerance
        else:
            return bound - objective > self.absolute_tolerance

    def bound_is_suboptimal(self, bound, objective):
        """Returns True when bound is worse than the
        objective by greater than the absolute tolerance."""
        # handles the both equal and infinite case
        if bound == objective:
            return False
        if self.sense == minimize:
            return bound - objective > self.absolute_tolerance
        else:
            return objective - bound > self.absolute_tolerance

    def worst_bound(self, *bounds):
        """Returns the worst bound, as defined by the
        objective sense, from a given iterable of bound
        values."""
        if self.sense == minimize:
            return min(*bounds)
        else:
            return max(*bounds)

    def best_bound(self, *bounds):
        """Returns the best bound, as defined by the
        objective sense, from a given iterable of bound
        values."""
        if self.sense == minimize:
            return max(*bounds)
        else:
            return min(*bounds)

    def worst_objective(self, *objectives):
        """Returns the worst objective, as defined by the
        objective sense, from a given iterable of objective
        values."""
        if self.sense == minimize:
            return max(*objectives)
        else:
            return min(*objectives)

    def best_objective(self, *objectives):
        """Returns the best objective, as defined by the
        objective sense, from a given iterable of objective
        values."""
        if self.sense == minimize:
            return min(*objectives)
        else:
            return max(*objectives)
