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

def _scale_absolute_gap(gap, scale):
    """Convert an absolute gap to a relative gap with the
    given scaling factor."""
    assert scale > 0
    if not math.isinf(gap):
        return gap / scale
    else:
        return gap

def _default_scale(bound, objective):
    """`max{1.0,|objective|}`"""
    # avoid using abs() as it uses the same logic, but adds
    # additional function call overhead
    if objective > 1.0:
        return objective
    elif objective < -1.0:
        return -objective
    else:
        return 1.0

def compute_relative_gap(sense,
                         bound,
                         objective,
                         scale=_default_scale):
    """Returns the relative gap between the bound and
    the objective, respecting the sign relative to the
    objective sense of this problem."""
    return _scale_absolute_gap(compute_absolute_gap(sense,
                                                    bound,
                                                    objective),
                               scale(bound, objective))

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
    scale_function : function, optional
        A function with signature `f(bound, objective) ->
        float` that returns a positive scale factor used to
        convert the absolute difference between the bound
        and objective into a relative difference. The
        relative difference is compared with the
        `relative_gap` convergence tolerance to determine if
        the solver should terminate. The default is
        equivalent to `max{1.0,|objective|}`. Other examples
        one could use are `max{|bound|,|objective|}`,
        `(|bound|+|objective|)/2`, etc.
    queue_tolerance : float, optional
        The absolute tolerance used when deciding if a node
        is eligible to enter the queue. The difference
        between the node bound and the incumbent objective
        must be greater than this value. The default setting
        of zero means that nodes whose bound is equal to the
        incumbent objective are not eligible to enter the
        queue. Setting this to larger values can be used to
        control the queue size, but it should be kept small
        enough to allow absolute and relative optimality
        tolerances to be met. This option can also be set to
        `None` to allow nodes with a bound equal to (but not
        greater than) the incumbent objective to enter the
        queue. (default: 0)
    branch_tolerance : float, optional
        The absolute tolerance used when deciding if the
        computed objective and bound for a node are
        sufficiently different to branch into the node. The
        default value of zero means that branching will
        occur if the bound is not exactly equal to the
        objective. This option can be set to `None` to
        enable branching for nodes with a bound and
        objective that are exactly equal. (default: 0)
    comparison_tolerance : float, optional
        The absolute tolerance used when deciding if two
        objective or bound values are sufficiently different
        to be considered improved or worsened. This
        tolerance controls when the solver considers a new
        incumbent objective to be found. It also controls
        when warnings are output about bounds becoming worse
        on child nodes. Setting this to larger values can be
        used to avoid the above solver actions due to
        insignificant numerical differences, but it is
        better to deal with these numerical issues by
        rounding numbers to a reliable precision before
        returning them from the problem methods.
        (default: 0)
    objective_stop : float, optional
        If provided, the "objective_limit" termination
        criteria is met when a feasible objective is found
        that is at least as good as the specified value. If
        this value is infinite, the termination criteria is
        met as soon as a finite objective is found.
        (default: None)
    bound_stop : float, optional
        If provided, the "objective_limit" termination
        criteria is met when the best bound on the objective
        is at least as good as the specified value. If this
        value is infinite, the termination criteria is met
        as soon as a finite objective is found.
        (default: None)
    """
    __slots__ = ("sense",
                 "absolute_gap",
                 "relative_gap",
                 "scale_function",
                 "queue_tolerance",
                 "branch_tolerance",
                 "comparison_tolerance",
                 "objective_stop",
                 "bound_stop",
                 "infeasible_objective",
                 "unbounded_objective")

    def __init__(self,
                 sense,
                 absolute_gap=1e-8,
                 relative_gap=1e-4,
                 scale_function=_default_scale,
                 queue_tolerance=0,
                 branch_tolerance=0,
                 comparison_tolerance=0,
                 objective_stop=None,
                 bound_stop=None):
        self.sense = sense
        if self.sense == minimize:
            self.infeasible_objective = inf
            self.unbounded_objective = -inf
        else:
            assert self.sense == maximize
            self.infeasible_objective = -inf
            self.unbounded_objective = inf
        self.absolute_gap = None
        if absolute_gap is not None:
            self.absolute_gap = float(absolute_gap)
            assert (self.absolute_gap >= 0) and \
                (not math.isinf(self.absolute_gap))
        self.relative_gap = None
        if relative_gap is not None:
            self.relative_gap = float(relative_gap)
            assert self.relative_gap >= 0 and \
                (not math.isinf(self.relative_gap))
        self.scale_function = scale_function
        self.queue_tolerance = float(queue_tolerance) \
            if (queue_tolerance is not None) else queue_tolerance
        assert (self.queue_tolerance is None) or \
            ((self.queue_tolerance >= 0) and \
            (not math.isinf(self.queue_tolerance)) and \
            (not math.isnan(self.queue_tolerance)))
        self.branch_tolerance = float(branch_tolerance) \
            if (branch_tolerance is not None) else branch_tolerance
        assert (self.branch_tolerance is None) or \
            ((self.branch_tolerance >= 0) and \
            (not math.isinf(self.branch_tolerance)) and \
            (not math.isnan(self.branch_tolerance)))
        self.comparison_tolerance = float(comparison_tolerance)
        assert self.comparison_tolerance >= 0 and \
            (not math.isinf(self.comparison_tolerance)) and \
            (not math.isnan(self.comparison_tolerance))
        self.objective_stop = None
        if objective_stop is not None:
            self.objective_stop = float(objective_stop)
            assert self.objective_stop != self.unbounded_objective
            assert not math.isnan(self.objective_stop)
        self.bound_stop = None
        if bound_stop is not None:
            self.bound_stop = float(bound_stop)
            assert self.bound_stop != self.infeasible_objective
            assert not math.isnan(self.bound_stop)

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
            if self.objective_stop != self.infeasible_objective:
                if self.sense == minimize:
                    if best_objective <= self.objective_stop:
                        result = TerminationCondition.objective_limit
                else:
                    if best_objective >= self.objective_stop:
                        result = TerminationCondition.objective_limit
            else:
                if best_objective != self.infeasible_objective:
                    result = TerminationCondition.objective_limit
        elif self.bound_stop is not None:
            if self.bound_stop != self.unbounded_objective:
                if self.sense == minimize:
                    if global_bound >= self.bound_stop:
                        result = TerminationCondition.objective_limit
                else:
                    if global_bound <= self.bound_stop:
                        result = TerminationCondition.objective_limit
            else:
                if global_bound != self.unbounded_objective:
                    result = TerminationCondition.objective_limit
        return result

    def objective_is_optimal(self, objective, bound):
        """Determines if the objective is optimal by
        checking if the optimality gap is small enough
        relative to the absolute gap or relative gap
        settings."""
        assert bound != self.infeasible_objective
        if (objective != self.unbounded_objective) and \
           (objective != self.infeasible_objective):
            gap = self.compute_absolute_gap(bound,
                                            objective)
            if (self.absolute_gap is not None) and \
               (gap <= self.absolute_gap):
                return True
            elif (self.relative_gap is not None):
                scale = self.scale_function(bound, objective)
                gap = _scale_absolute_gap(gap, scale)
                if gap <= self.relative_gap:
                    return True
        return False

    def compute_absolute_gap(self, bound, objective):
        """Returns the absolute gap between the bound and
        the objective, respecting the sign relative to the
        objective sense of this problem."""
        return compute_absolute_gap(self.sense,
                                    bound,
                                    objective)

    def compute_relative_gap(self, bound, objective):
        """Returns the relative gap between the bound and
        the objective, respecting the sign relative to the
        objective sense of this problem."""
        return compute_relative_gap(self.sense,
                                    bound,
                                    objective,
                                    scale=self.scale_function)

    def eligible_for_queue(self, bound, objective):
        """Returns True when the queue object with the given
        bound is eligible for the queue relative to the
        given objective."""
        if (bound == self.infeasible_objective) or \
           (objective == self.unbounded_objective):
            return False
        if self.sense == minimize:
            delta = objective - bound
        else:
            delta = bound - objective
        assert not math.isnan(delta)
        if self.queue_tolerance is not None:
            return delta > self.queue_tolerance
        else:
            return delta >= 0

    def eligible_to_branch(self, bound, objective):
        """Returns True when the bound and objective
        are sufficiently far apart to allow branching."""
        if (objective == self.unbounded_objective) or \
           (bound == self.infeasible_objective):
            return False
        if self.sense == minimize:
            delta = objective - bound
        else:
            delta = bound - objective
        assert not math.isnan(delta)
        if self.branch_tolerance is not None:
            return delta > self.branch_tolerance
        else:
            return delta >= 0

    def bound_worsened(self, new, old):
        """Returns True when the new bound is worse than the
        old bound by greater than the comparison
        tolerance."""
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            delta = old - new
        else:
            delta = new - old
        assert not math.isnan(delta)
        return delta > self.comparison_tolerance

    def objective_improved(self, new, old):
        """Returns True when the new objective is better
        than the old objective by greater than the
        comparison tolerance."""
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            delta = old - new
        else:
            delta = new - old
        assert not math.isnan(delta)
        return delta > self.comparison_tolerance

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
