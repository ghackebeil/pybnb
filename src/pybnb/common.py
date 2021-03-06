"""
Basic definitions and utilities.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

from typing import List, Dict
import enum


@enum.unique
class ProblemSense(enum.IntEnum):
    minimize = 1  # type: int
    """The objective sense defining a minimization problem."""

    maximize = -1  # type: int
    """The objective sense defining a maximization problem."""


minimize = ProblemSense.minimize
maximize = ProblemSense.maximize

inf = float("inf")  # type: float
"""A floating point constant set to ``float('inf')``."""

nan = float("nan")  # type: float
"""A floating point constant set to ``float('nan')``."""


@enum.unique
class QueueStrategy(str, enum.Enum):
    """Strategies for prioritizing nodes in the central
    dispatcher queue. For all strategies, ties are broken by
    insertion order."""

    bound = "bound"
    """The node with the worst bound is always selected
    next."""
    objective = "objective"
    """The node with the best objective is always selected
    next."""
    breadth = "breadth"
    """The node with the smallest tree depth is always
    selected next (i.e., breadth-first search)."""
    depth = "depth"
    """The node with the largest tree depth is always
    selected next (i.e., depth-first search)."""
    local_gap = "local_gap"
    """The node with the largest gap between its local
    objective and bound is always selected next."""
    fifo = "fifo"
    """Nodes are served in first-in, first-out order."""
    lifo = "lifo"
    """Nodes are served in last-in, first-out order."""
    random = "random"
    """Nodes are assigned a random priority before entering
    the queue."""
    custom = "custom"
    """The node with the largest value stored in the
    :attr:`queue_priority <pybnb.node.Node.queue_priority>`
    attribute is always selected next. Users are expected to
    assign a priority to all nodes returned from the
    :func:`branch <pybnb.problem.Problem.branch>`
    method on their problem."""


_queue_strategy_to_int = {}  # type: Dict[QueueStrategy, int]
_int_to_queue_strategy = []  # type: List[QueueStrategy]
for _i_qs, _val_qs in enumerate(sorted(QueueStrategy, key=lambda v: v.value)):
    _queue_strategy_to_int[_val_qs] = _i_qs
    _int_to_queue_strategy.append(_val_qs)
del _i_qs
del _val_qs


@enum.unique
class SolutionStatus(str, enum.Enum):
    """Possible values assigned to the
    :attr:`solution_status` attribute of a
    :class:`SolverResults <pybnb.solver_results.SolverResults>`
    object returned from a solve."""

    optimal = "optimal"
    """Indicates that the best objective is finite and close
    enough to the global bound to satisfy the optimality
    tolerances used for the solve."""
    feasible = "feasible"
    """Indicates that the best objective is finite but not
    close enough to the global bound to satisfy the
    optimality tolerances used for the solve."""
    infeasible = "infeasible"
    """Indicates that both the best objective and global
    bound are equal to the infeasible objective value (+inf
    or -inf depending on the sense)."""
    unbounded = "unbounded"
    """Indicates that both the best objective and global
    bound are equal to the unbounded objective value (+inf
    or -inf depending on the sense)."""
    invalid = "invalid"
    """Indicates that the global bound is not a valid bound
    on the best objective found. This may be due to an
    ill-defined problem or other numerical issues."""
    unknown = "unknown"
    """Indicates that the global bound is finite, but no
    feasible (finite) objective was found."""


_solution_status_to_int = {}  # type: Dict[SolutionStatus, int]
_int_to_solution_status = []  # type: List[SolutionStatus]
for _i_ss, _val_ss in enumerate(sorted(SolutionStatus, key=lambda v: v.value)):
    _solution_status_to_int[_val_ss] = _i_ss
    _int_to_solution_status.append(_val_ss)
del _i_ss
del _val_ss


@enum.unique
class TerminationCondition(str, enum.Enum):
    """Possible values assigned to the
    :attr:`termination_condition` attribute of a
    :class:`SolverResults <pybnb.solver_results.SolverResults>`
    object returned from a solve."""

    optimality = "optimality"
    """The dispatcher terminated the solve based on
    optimality criteria."""
    objective_limit = "objective_limit"
    """The dispatcher terminated the solve based on the
    user-supplied limit on the objective or bound being
    satisfied."""
    node_limit = "node_limit"
    """The dispatcher terminated the solve due to the
    user-supplied explored node limit being surpassed."""
    time_limit = "time_limit"
    """The dispatcher terminated the solve due to the
    user-supplied time limit being surpassed."""
    queue_empty = "queue_empty"
    """The dispatcher terminated the solve due to the node
    queue becoming empty."""
    queue_limit = "queue_limit"
    """The dispatcher terminated the solve due to the
    user-supplied queue size limit being exceeded."""
    interrupted = "interrupted"
    """Solve termination was initiated by SIGINT or SIGUSR
    signal event."""


_termination_condition_to_int = {}  # type: Dict[TerminationCondition, int]
_int_to_termination_condition = []  # type: List[TerminationCondition]
for _i_tc, _val_tc in enumerate(sorted(TerminationCondition, key=lambda v: v.value)):
    _termination_condition_to_int[_val_tc] = _i_tc
    _int_to_termination_condition.append(_val_tc)
del _i_tc
del _val_tc
