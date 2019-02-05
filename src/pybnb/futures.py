import copy
import logging
import math

from pybnb.node import Node
from pybnb.dispatcher import (DispatcherQueueData,
                              DispatcherLocal)
from pybnb.misc import get_simple_logger
from pybnb.solver import Solver
from pybnb.problem import _ProblemWithSolveInfoCollection

class _RedirectHandler(logging.Handler):

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

class nested_solve(_ProblemWithSolveInfoCollection):

    def __init__(self, problem, **kwds):
        self._problem = problem
        self._solver = Solver(comm=None)
        self._solve_options = kwds
        self._log = logging.Logger(None,
                                   level=logging.WARNING)
        self._results = None
        self._children = None
        self._current_node = None
        self._solve_found_best_node = False
        self._best_objective = None
        self._best_node = None
        self._convergence_checker = None
        super(nested_solve, self).__init__()

    def _initialize(self, dispatcher, best_objective):
        assert best_objective is not None
        self._best_objective = best_objective
        self._log.addHandler(_RedirectHandler(dispatcher))

    def _solve(self):
        bound_stop = None
        if not math.isinf(self._best_objective):
            bound_stop = self._best_objective
        # shallow copy
        root = copy.copy(self._current_node)
        queue = DispatcherQueueData(
            nodes=[root],
            worst_terminal_bound=None,
            sense=self._convergence_checker.sense)
        self._results = self._solver.solve(
            self._problem,
            best_objective=self._best_objective,
            best_node=self._best_node,
            bound_stop=bound_stop,
            initialize_queue=queue,
            log=self._log,
            **self._solve_options)
        if (self._results.best_node is not None) and \
           ((self._best_node is None) or \
            (self._best_node._uuid != self._results.best_node._uuid)):
            self._best_node = self._results.best_node
            self._best_objective = self._best_node.objective
            self._solve_found_best_node = True
        queue = self._solver.save_dispatcher_queue()
        self._children = queue.nodes
        if self._solve_found_best_node:
            for child in self._children:
                if child is self._best_node:
                    break
            else: # no break in for-loop
                # add the best node to the children so it will
                # make it to the top-level dispatcher
                self._yield_best_node = True
        self._solve_info.add_from(
            self._solver._global_solve_info)

    #
    # Define Problem methods
    #

    def sense(self):
        return self._problem.sense()

    def objective(self):
        assert self._results is not None
        if (self._results.objective != \
            self._convergence_checker.unbounded_objective) and \
            self._convergence_checker.eligible_for_queue(
                self._results.bound,
                self._best_objective) and \
            self._convergence_checker.eligible_to_branch(
                self._results.bound,
                self._results.objective) and \
            self._solve_found_best_node:
            # we will be returning the new best node as a child
            # so collect the best objective there
            return self._current_node.objective
        else:
            return self._results.objective

    def bound(self):
        # after a new state is loaded, bound MUST be called
        # before any of the other methods
        if self._results is None:
            self._solve()
        return self._results.bound

    def branch(self):
        assert self._results is not None
        if self._yield_best_node:
            yield self._best_node
        for child in self._children:
            yield child

    def save_state(self, node):
        self._problem.save_state(node)

    def load_state(self, node):
        self._problem.load_state(node)
        self._results = None
        self._children = None
        self._current_node = node
        self._solve_found_best_node = False
        self._yield_best_node = False

    def notify_solve_begins(self,
                            comm,
                            worker_comm,
                            convergence_checker):
        self._results = None
        self._children = None
        self._current_node = None
        self._solve_found_best_node = False
        self._best_objective = None
        self._best_node = None
        self._convergence_checker = convergence_checker
        self._problem.notify_solve_begins(
            comm,
            worker_comm,
            convergence_checker)

    def notify_new_best_node(self, node, current):
        self._best_node = node
        self._best_objective = node.objective
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
