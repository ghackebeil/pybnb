import copy
import logging
import math

from pybnb.dispatcher import DispatcherQueueData
from pybnb.solver import Solver
from pybnb.problem import _ProblemWithSolveInfoCollection

class _RedirectHandler(logging.Handler):
    """Redirects log messages with a WARNING level or above
    to the provided dispatcher's log."""

    def __init__(self, dispatcher):
        super(_RedirectHandler, self).__init__()
        self._dispatcher = dispatcher

    def emit(self, record):
        if logging.WARNING <= record.levelno < logging.ERROR:
            self._dispatcher.log_warning(record.getMessage())
        elif logging.ERROR <= record.levelno < logging.CRITICAL:
            self._dispatcher.log_error(record.getMessage())
        elif logging.CRITICAL <= record.levelno:
            self._dispatcher.log_critical(record.getMessage())

class NestedSolver(_ProblemWithSolveInfoCollection):
    """A wrapper for problem implementations that uses a
    nested branch-and-bound solve to process a node.

    Parameters
    ----------
    problem : :class:`pybnb.Problem <pybnb.problem.Problem>`
        An object defining a branch-and-bound problem.
    node_limit : int, optional
        The same as the standard solver option, but applied
        to the nested solver to limit the number nodes to
        explore when processing a work item. (default: None)
    time_limit : float, optional
        The same as the standard solver option, but applied
        to the nested solver to limit the amount of time
        spent processing a work item. (default: 5)
    queue_strategy : :class:`QueueStrategy <pybnb.common.QueueStrategy>` or tuple
        The same as the standard solver option, but applied
        to the nested solver to control the queue strategy
        used when processing a work item. (default: 'depth')
    """

    def __init__(self,
                 problem,
                 node_limit=None,
                 time_limit=5,
                 queue_strategy="depth"):
        self._problem = problem
        self._node_limit = node_limit
        self._time_limit = time_limit
        self._queue_strategy = queue_strategy
        self._solver = Solver(comm=None)
        self._log = logging.Logger(None,
                                   level=logging.WARNING)
        self._convergence_checker = None
        self._best_objective = None
        self._best_node = None
        self._current_node = None
        self._results = None
        self._queue = None
        super(NestedSolver, self).__init__()

    def _initialize(self,
                    dispatcher,
                    best_objective,
                    disable_objective_call):
        assert best_objective is not None
        self._best_objective = best_objective
        self._disable_objective_call = disable_objective_call
        self._log.addHandler(_RedirectHandler(dispatcher))

    def _solve(self):
        bound_stop = None
        if not math.isinf(self._best_objective):
            bound_stop = self._best_objective
        # shallow copy
        root = copy.copy(self._current_node)
        init_queue = DispatcherQueueData(
            nodes=[root],
            worst_terminal_bound=None,
            sense=self._convergence_checker.sense)
        self._results = self._solver.solve(
            self._problem,
            best_objective=self._best_objective,
            best_node=self._best_node,
            bound_stop=bound_stop,
            initialize_queue=init_queue,
            log=self._log,
            absolute_gap=\
                self._convergence_checker.absolute_gap,
            relative_gap=\
                self._convergence_checker.relative_gap,
            scale_function=\
                self._convergence_checker.scale_function,
            queue_tolerance=\
                self._convergence_checker.queue_tolerance,
            branch_tolerance=\
                self._convergence_checker.branch_tolerance,
            comparison_tolerance=\
                self._convergence_checker.comparison_tolerance,
            objective_stop=\
                self._convergence_checker.objective_stop,
            disable_objective_call=self._disable_objective_call,
            node_limit=self._node_limit,
            time_limit=self._time_limit,
            queue_strategy=self._queue_strategy,
            disable_signal_handlers=True)
        self._queue = self._solver.save_dispatcher_queue()
        self._solve_info.add_from(
            self._solver._global_solve_info)

    #
    # Ducktype a partial Problem interface
    #

    def sense(self):
        return self._problem.sense()

    def save_state(self, node):
        self._problem.save_state(node)

    def load_state(self, node):
        self._problem.load_state(node)
        self._results = None
        self._children = None
        self._current_node = node
        self._solve_found_best_node = False

    def notify_solve_begins(self,
                            comm,
                            worker_comm,
                            convergence_checker):
        self._best_objective = None
        self._best_node = None
        self._convergence_checker = convergence_checker
        self._current_node = None
        self._results = None
        self._queue = None
        self._problem.notify_solve_begins(
            comm,
            worker_comm,
            convergence_checker)

    def notify_new_best_node(self, node, current):
        self._best_objective = self._convergence_checker.\
            best_objective(self._best_objective,
                           node.objective)
        self._best_node = node
        self._problem.notify_new_best_node(node, current)

    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        while len(self._log.handlers) > 0:
            self._log.removeHandler(self._log.handlers[0])
        self._problem.notify_solve_finished(
            comm,
            worker_comm,
            results)
