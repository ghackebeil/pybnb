"""
Branch-and-bound dispatcher implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import array
import time
import array
import collections
import os
import socket
import logging
import marshal
import math

from pybnb.configuration import config
from pybnb.common import (minimize,
                          maximize,
                          inf,
                          TerminationCondition,
                          _termination_condition_to_int)
from pybnb.misc import get_gap_labels
from pybnb.dispatcher_proxy import (ProcessType,
                                    DispatcherAction,
                                    DispatcherResponse,
                                    DispatcherProxy)
from pybnb.node import _SerializedNode
from pybnb.problem import _SolveInfo
from pybnb.mpi_utils import Message
from pybnb.priority_queue import PriorityQueueFactory

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

from sortedcontainers import SortedList

import six

class DispatcherQueueData(object):
    """A namedtuple storing data that can be used to
    initialize a dispatcher queue.

    Attributes
    ----------
    nodes : list
        A list of :class:`Node <pybnb.node.Node>` objects.
    worst_terminal_bound : float or None
        The worst bound of any node where branching did not
        continue.
    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem that produced
        this queue.
    """
    __slots__ = ("nodes",
                 "worst_terminal_bound",
                 "sense")
    def __init__(self,
                 nodes,
                 worst_terminal_bound,
                 sense):
        self.nodes = nodes
        self.worst_terminal_bound = worst_terminal_bound
        self.sense = sense
        assert sense in (minimize, maximize)

    def bound(self):
        """Returns the global bound defined by this queue data."""
        bound = None
        if len(self.nodes) > 0:
            if self.sense == minimize:
                bound = min(n_.bound for n_ in self.nodes)
            else:
                assert self.sense == maximize
                bound = max(n_.bound for n_ in self.nodes)
        if self.worst_terminal_bound is not None:
            if bound is not None:
                if self.sense == minimize:
                    bound = min(bound,
                                self.worst_terminal_bound)
                else:
                    assert self.sense == maximize
                    bound = max(bound,
                                self.worst_terminal_bound)
            else:
                bound = self.worst_terminal_bound
        return bound

class StatusPrinter(object):
    """Logs status information about the branch-and-bound
    solve.

    Parameters
    ----------
    dispatcher : :class:`pybnb.dispatcher.Dispatcher`
        The central dispatcher that will be monitored.
    log : :class:`logging.Logger`
        A log object where solver output should be sent.
    log_interval_seconds : float
        The approximate maximum time (in seconds) between
        solver log updates. More time may pass between log
        updates if no updates have been received from any
        workers, and less time may pass if a new incumbent
        is found. (default: 1.0)
    """

    def __init__(self,
                 dispatcher,
                 log,
                 log_interval_seconds=1.0):
        assert log_interval_seconds >= 0
        self._dispatcher = dispatcher
        self._log_interval_seconds = log_interval_seconds
        self._log = log

        percent_relative_gap_tol = 1e-6
        if (self._dispatcher.converger.relative_gap is not None) and \
           self._dispatcher.converger.relative_gap != 0:
            percent_relative_gap_tol = 100.0 * \
                self._dispatcher.converger.relative_gap
        rgap_str_length, rgap_label_str, rgap_number_str = \
            get_gap_labels(percent_relative_gap_tol, key="rgap")

        absolute_gap_tol = 1e-8
        if (self._dispatcher.converger.absolute_gap is not None) and \
           self._dispatcher.converger.absolute_gap != 0:
            absolute_gap_tol = \
                self._dispatcher.converger.absolute_gap
        agap_str_length, agap_label_str, agap_number_str = \
            get_gap_labels(absolute_gap_tol, key="agap", format='g')

        assert rgap_str_length >= 10
        assert agap_str_length >= 10
        extra_space = (rgap_str_length-10) + (agap_str_length-10)
        extra_space_left = extra_space // 2
        extra_space_right = (extra_space // 2) + (extra_space % 2)
        self._lines = ("--------------------"
                       "--------------------"
                       "--------------------"
                       "--------------------"
                       "--------------------"
                       "----------------")+("-"*extra_space)
        self._initial_header_line = \
            (self._lines + "\n"
             "         Nodes        |" + \
             (" "*extra_space_left) + \
             "                   Objective Bounds                    " + \
             (" "*extra_space_right) + \
             "|              Work              ")
        self._header_line = \
            (" {explored:>9} {unexplored:>9}  |{objective:>15} "
             "{bound:>15} "+rgap_label_str+"  "+agap_label_str+" |{runtime:>9} {rate:>10} "
             "{imbalance:>9}  {idle:>5}").\
             format(explored="Expl",
                    unexplored="Unexpl",
                    objective="Incumbent",
                    bound="Bound",
                    runtime="Time (s)",
                    rgap="Rel. Gap",
                    agap="Abs. Gap",
                    rate="Nodes/Sec",
                    imbalance="Imbalance",
                    idle="Idle")
        self._line_template = \
            ("{tag:>1}{explored:>9d} {unexplored:>9d}  |{objective:>15.7g} "
             "{bound:>15.7g} "+rgap_number_str+"% "+agap_number_str+" |{runtime:>9.1f} {rate:>10.2f} "
             "{imbalance:>8.2f}% {idle:>6d}")
        self._line_template_big_gap = \
            ("{tag:>1}{explored:>9d} {unexplored:>9d}  |{objective:>15.7g} "
             "{bound:>15.7g} "+rgap_label_str+"% "+agap_number_str+" |{runtime:>9.1f} {rate:>10.2f} "
             "{imbalance:>8.2f}% {idle:>6d}")

        self._last_print_time = float('-inf')
        served, explored, unexplored = self._dispatcher._get_node_counts()
        assert served == 0
        assert explored == 0
        assert unexplored == 0
        self._last_explored_nodes_count = 0
        self._last_rgap = None
        self._smoothing = 0.95
        self._avg_time_per_node = None
        self._print_count = 0
        self._new_objective = False
        self._report_new_objective = False

    def log_info(self, msg):
        """Pass a message to ``log.info``"""
        self._log.info(msg)

    def log_warning(self, msg):
        """Pass a message to ``log.warning``"""
        self._log.warning(msg)

    def log_debug(self, msg):
        """Pass a message to ``log.debug``"""
        self._log.debug(msg)

    def log_error(self, msg):
        """Pass a message to ``log.error``"""
        self._log.error(msg)

    def log_critical(self, msg):
        """Pass a message to ``log.critical``"""
        self._log.critical(msg)

    def new_objective(self, report=True):
        """Indicate that a new objective has been found

        Parameters
        ----------
        report : bool, optional
            Indicate whether or not to force the next `tic`
            log output. (default: False)
        """
        self._new_objective = True
        self._report_new_objective = report

    def tic(self, force=False):
        """Provide an opportunity to log output if certain
        criteria are met.

        Parameters
        ----------
        force : bool, optional
            Indicate whether or not to force logging of
            output, even if logging criteria are not
            met. (default: False)
        """
        if not self._log.isEnabledFor(logging.INFO):
            return
        current_time = self._dispatcher.clock()
        new_objective = self._new_objective
        report_new_objective = self._report_new_objective
        delta_t = current_time - self._last_print_time
        self._report_new_objective = False
        if not force:
            if not report_new_objective:
                if delta_t < self._log_interval_seconds:
                    return
        self._new_objective = False
        (served_nodes_count,
         explored_nodes_count,
         unexplored_nodes_count) = self._dispatcher._get_node_counts()
        delta_n = explored_nodes_count - \
                  self._last_explored_nodes_count
        if delta_t and delta_n:
            if self._avg_time_per_node is None:
                self._avg_time_per_node = delta_t / float(delta_n)
            else:
                self._avg_time_per_node = \
                    self._smoothing * delta_t / float(delta_n) + \
                    (1 - self._smoothing) * self._avg_time_per_node
        if self._avg_time_per_node:
            rate = 1.0/self._avg_time_per_node
        else:
            rate = 0.0

        imbalance = self._dispatcher._compute_load_imbalance()
        idle = 0
        if hasattr(self._dispatcher,"needs_work_queue"):
            idle = len(self._dispatcher.needs_work_queue)

        tag = '' if (not new_objective) else '*'
        bound = self._dispatcher._get_current_bound()
        objective = self._dispatcher.best_objective
        agap = self._dispatcher.converger.\
               compute_absolute_gap(bound, objective)
        rgap = self._dispatcher.converger.\
               compute_relative_gap(bound, objective)
        rgap *= 100.0
        if (self._print_count % 5) == 0:
            if self._print_count == 0:
                self._log.info(self._initial_header_line)
            self._log.info(self._header_line)

        if (rgap != inf) and \
           (rgap > 9999.0):
            self._log.info(self._line_template_big_gap.format(
                tag=tag,
                explored=explored_nodes_count,
                unexplored=unexplored_nodes_count,
                objective=objective,
                bound=bound,
                rgap="9999+",
                agap=agap,
                runtime=current_time-self._dispatcher.start_time,
                rate=rate,
                imbalance=imbalance,
                idle=idle))
            self._last_rgap = None
        else:
            self._log.info(self._line_template.format(
                tag=tag,
                explored=explored_nodes_count,
                unexplored=unexplored_nodes_count,
                objective=objective,
                bound=bound,
                rgap=rgap,
                agap=agap,
                runtime=current_time-self._dispatcher.start_time,
                rate=rate,
                imbalance=imbalance,
                idle=idle))
            self._last_rgap = rgap
        self._last_explored_nodes_count = explored_nodes_count
        self._print_count += 1
        self._last_print_time = current_time

class DispatcherBase(object):
    """The base dispatcher implementation with some core
    functionality shared by the distributed and local
    implementations."""

    def __init__(self):
        self.start_time = None
        self.initialized = False
        self.log_new_incumbent = False
        self.termination_condition = None
        self.converger = None
        self.last_global_bound = None
        self.track_bound = False
        self.queue = None
        self.best_node = None
        self.best_objective = None
        self.journalist = None
        self.node_limit = None
        self.time_limit = None
        self.queue_limit = None
        self.served_nodes_count = None
        self.worst_terminal_bound = None
        self.clock = None

    def _add_work_to_queue(self, node):
        bound = node.bound
        if self.converger.eligible_for_queue(
                bound,
                self.best_objective):
            self.queue.put(node)
            if (self.queue_limit is not None) and \
               (self.queue.size() > self.queue_limit) and \
               (self.termination_condition is None):
                self.termination_condition = \
                    TerminationCondition.queue_limit
            return True
        else:
            self._check_update_worst_terminal_bound(bound)
            return False

    def _check_update_worst_terminal_bound(self, bound):
        if (self.worst_terminal_bound is None) or \
           self.converger.bound_worsened(bound,
                                         self.worst_terminal_bound):
            self.worst_terminal_bound = bound

    def _check_update_best_objective(self, objective):
        assert objective is not None
        assert not math.isnan(objective)
        updated = False
        if self.converger.objective_improved(
                objective,
                self.best_objective):
            self.best_objective = objective
            updated = True
            if self.journalist is not None:
                self.journalist.new_objective(
                    report=self.log_new_incumbent)
            eligible_for_queue_ = self.converger.eligible_for_queue
            removed = self.queue.filter(
                lambda node_: eligible_for_queue_(
                    node_.bound,
                    self.best_objective))
            for node_ in removed:
                self._check_update_worst_terminal_bound(node_.bound)
        return updated

    def _check_update_best_node(self, node):
        best_objective_updated = False
        objective = node.objective
        assert objective is not None
        assert not math.isnan(objective)
        if (objective != self.converger.infeasible_objective) and \
           ((self.best_node is None) or \
            self.converger.objective_improved(
                objective,
                self.best_node.objective)):
            assert node._uuid is not None
            self.best_node = node
            best_objective_updated = \
                self._check_update_best_objective(objective)
        return best_objective_updated

    def _check_termination(self):
        # check if we are done
        if self.termination_condition is None:
            if self.track_bound:
                global_bound = self._get_current_bound()
                self.last_global_bound = global_bound
                self.termination_condition = self.converger.\
                    check_termination_criteria(global_bound,
                                               self.best_objective)
            if self.termination_condition is None:
                if self.node_limit is not None:
                    (served_nodes_count, explored_nodes_count, _) = \
                        self._get_node_counts()
                    # we need to check the explored count as
                    # it may be greater than the served
                    # count due to the use of a nested solver
                    if (served_nodes_count >= self.node_limit) or \
                       (explored_nodes_count >= self.node_limit):
                        self.termination_condition = \
                            TerminationCondition.node_limit
                if self.termination_condition is None:
                    if (self.time_limit is not None) and \
                       ((self.clock() - self.start_time) >= self.time_limit):
                        self.termination_condition = \
                            TerminationCondition.time_limit

    def _get_work_item(self):
        node = self.queue.get()
        assert node is not None
        self.served_nodes_count += 1
        return node

    #
    # Interface
    #

    def initialize(self,
                   best_objective,
                   best_node,
                   initialize_queue,
                   queue_strategy,
                   converger,
                   node_limit,
                   time_limit,
                   queue_limit,
                   track_bound,
                   log,
                   log_interval_seconds,
                   log_new_incumbent):
        """Initialize the dispatcher for a new solve.

        Parameters
        ----------
        best_objective : float
            The assumed best objective to start with.
        best_node : :class:`Node <pybnb.node.Node>`
            A node storing the assumed best objective.
        initialize_queue : :class:`pybnb.dispatcher.DispatcherQueueData`
            The initial queue.
        queue_strategy : :class:`QueueStrategy <pybnb.common.QueueStrategy>`
            Sets the strategy for prioritizing nodes in the
            central dispatcher queue. See the
            :class:`QueueStrategy <pybnb.common.QueueStrategy>`
            enum for the list of acceptable values.
        converger : :class:`pybnb.convergence_checker.ConvergenceChecker`
            The branch-and-bound convergence checker object.
        node_limit : int or None
            An integer representing the maximum number of
            nodes to processes before beginning to terminate
            the solve. If None, no node limit will be
            enforced.
        time_limit : float or None
            The maximum amount of time to spend processing
            nodes before beginning to terminate the
            solve. If None, no time limit will be enforced.
        queue_limit : int or None
            The maximum allowed queue size. If exceeded, the
            solve will terminate. If None, no size limit on
            the queue will be enforced.
        log : ``logging.Logger``
            A log object where solver output should be sent.
        log_interval_seconds : float
            The approximate maximum time (in seconds)
            between solver log updates. More time may pass
            between log updates if no updates have been
            received from any workers, and less time may
            pass if a new incumbent is found.
        log_new_incumbent : bool
            Controls whether updates to the best objective
            are logged immediately (overriding the log
            interval). Setting this to false can be useful
            when frequent updates to the incumbent are
            expected and the additional logging slows down
            the dispatcher.
        """
        assert (node_limit is None) or \
            ((node_limit > 0) and \
             (node_limit == int(node_limit)))
        assert (time_limit is None) or \
            (time_limit >= 0)
        assert (queue_limit is None) or \
            (queue_limit >= 0)
        self.start_time = self.clock()
        self.initialized = True
        self.log_new_incumbent = log_new_incumbent
        self.termination_condition = None
        self.converger = converger
        if initialize_queue.sense != self.converger.sense:
            raise ValueError("The objective sense does not match "
                             "that of the initial queue.")
        self.last_global_bound = self.converger.unbounded_objective
        self.track_bound = track_bound
        self.queue = PriorityQueueFactory(
            queue_strategy,
            self.converger.sense,
            self.track_bound)
        self.best_objective = best_objective
        self.best_node = None
        if best_node is not None:
            assert best_node._uuid is not None
            self._check_update_best_node(best_node)
        self.node_limit = None
        if node_limit is not None:
            self.node_limit = int(node_limit)
        self.time_limit = None
        if time_limit is not None:
            self.time_limit = float(time_limit)
        self.queue_limit = None
        if queue_limit is not None:
            self.queue_limit = int(queue_limit)
        self.served_nodes_count = 0
        self.worst_terminal_bound = \
            initialize_queue.worst_terminal_bound
        self.journalist = None
        if (log is not None) and (not log.disabled):
            self.journalist = StatusPrinter(
                self,
                log,
                log_interval_seconds=log_interval_seconds)
        for node in initialize_queue.nodes:
            self._add_work_to_queue(node)

    def log_info(self, msg):
        """Pass a message to ``log.info``"""
        if self.journalist is not None:
            self.journalist.log_info(msg)

    def log_warning(self, msg):
        """Pass a message to ``log.warning``"""
        if self.journalist is not None:
            self.journalist.log_warning(msg)

    def log_debug(self, msg):
        """Pass a message to ``log.debug``"""
        if self.journalist is not None:
            self.journalist.log_debug(msg)

    def log_error(self, msg):
        """Pass a message to ``log.error``"""
        if self.journalist is not None:
            self.journalist.log_error(msg)

    def log_critical(self, msg):
        """Pass a message to ``log.critical``"""
        if self.journalist is not None:
            self.journalist.log_critical(msg)

    def save_dispatcher_queue(self):
        """Saves the current dispatcher queue. The result can
        be used to re-initialize a solve.

        Returns
        -------
        queue_data : :class:`pybnb.dispatcher.DispatcherQueueData`
            An object storing information that can be used
            to re-initialize the dispatcher queue to its
            current state.
        """
        return DispatcherQueueData(
            nodes=list(self.queue.items()),
            worst_terminal_bound=self.worst_terminal_bound,
            sense=self.converger.sense)

    #
    # Abstract Methods
    #

    def update(self, *args, **kwds):              #pragma:nocover
        raise NotImplementedError

    def _compute_load_imbalance(self):            #pragma:nocover
        """Get the worker load imbalance."""
        raise NotImplementedError()

    def _get_current_bound(self):                 #pragma:nocover
        """Get the current global bound"""
        raise NotImplementedError()

    def _get_final_solve_info(self):              #pragma:nocover
        """Get the final solve information"""
        raise NotImplementedError()

    def _get_node_counts(self):                   #pragma:nocover
        """Get the served and explored node counts"""
        raise NotImplementedError()

class DispatcherLocal(DispatcherBase):
    """The central dispatcher for a serial branch-and-bound
    algorithm."""

    def __init__(self):
        super(DispatcherLocal, self).__init__()
        self.external_bound = None
        self.solve_info = _SolveInfo()
        self.active_nodes = 0
        self.clock = time.time

    def _compute_load_imbalance(self):
        """Get the worker load imbalance."""
        return 0.0

    def _get_current_bound(self):
        """Get the current global bound"""
        bound = self.queue.bound()
        if self.converger.sense == maximize:
            if (self.external_bound is not None) and \
               ((bound is None) or \
                (self.external_bound > bound)):
                bound = self.external_bound
            if (self.worst_terminal_bound is not None) and \
               ((bound is None) or \
                (self.worst_terminal_bound > bound)):
                bound = self.worst_terminal_bound
        else:
            if (self.external_bound is not None) and \
               ((bound is None) or \
                (self.external_bound < bound)):
                bound = self.external_bound
            if (self.worst_terminal_bound is not None) and \
               ((bound is None) or \
                (self.worst_terminal_bound < bound)):
                bound = self.worst_terminal_bound
        if bound is not None:
            return bound
        else:
            # likely means the dispatcher was initialized
            # with an empty queue
            return self.converger.unbounded_objective

    def _get_final_solve_info(self):
        """Get the final solve information"""
        solve_info = _SolveInfo()
        solve_info.data[:] = self.solve_info.data
        return solve_info

    def _get_node_counts(self):
        return (self.served_nodes_count,
                self.solve_info.explored_nodes_count,
                self.queue.size() + self.active_nodes)

    #
    # Overloaded base class methods
    #

    def _check_update_best_objective(self, objective):
        updated = super(DispatcherLocal, self).\
            _check_update_best_objective(objective)
        if updated:
            assert self.best_objective == objective
            if self.external_bound is not None:
                if not self.converger.eligible_for_queue(
                        self.external_bound,
                        objective):
                    self.external_bound = None

    #
    # Interface
    #

    def initialize(self,
                   best_objective,
                   best_node,
                   initialize_queue,
                   queue_strategy,
                   converger,
                   node_limit,
                   time_limit,
                   queue_limit,
                   track_bound,
                   log,
                   log_interval_seconds,
                   log_new_incumbent):
        """Initialize the dispatcher. See the
        :func:`pybnb.dispatcher.DispatcherBase.initialize`
        method for argument descriptions."""
        self.solve_info.reset()
        self.active_nodes = 0
        super(DispatcherLocal, self).initialize(
            best_objective,
            best_node,
            initialize_queue,
            queue_strategy,
            converger,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        if self.journalist is not None:
            self.log_info("Starting branch & bound solve:\n"
                          " - dispatcher pid: %s (%s)\n"
                          " - worker processes: 1"
                          % (os.getpid(),
                             socket.gethostname()))
            self.journalist.tic()

    def update(self,
               best_objective,
               best_node,
               terminal_bound,
               solve_info,
               node_list):
        """Update local worker information.

        Parameters
        ----------
        best_objective : float or None
            A new potential best objective found by the
            worker.
        best_node : :class:`Node <pybnb.node.Node>` or None
            A new potential best node found by the worker.
        terminal_bound : float or None
            The worst bound of any terminal nodes that were
            processed by the worker since the last update.
        solve_info : :class:`_SolveInfo`
            The most up-to-date worker solve information.
        node_list : list
            A list of nodes to add to the queue.

        Returns
        -------
        solve_finished : bool
            Indicates if the dispatcher has terminated the solve.
        new_objective : float
            The best objective known to the dispatcher.
        best_node : :class:`Node <pybnb.node.Node>` or None
            The best node known to the dispatcher.
        data : :class:`Node <pybnb.node.Node>` or None
            If solve_finished is false, a new node for the
            worker to process. Otherwise, a tuple containing
            the global bound, the termination condition
            string, and the number of explored nodes.
        """
        assert self.initialized
        if best_objective is not None:
            self._check_update_best_objective(
                best_objective)
        if best_node is not None:
            assert best_node._uuid is not None
            self._check_update_best_node(best_node)
        self.solve_info.data[:] = solve_info.data
        self.external_bound = None
        self.active_nodes = 0
        if len(node_list) > 0:
            for node in node_list:
                self._add_work_to_queue(node)
        if terminal_bound is not None:
            self._check_update_worst_terminal_bound(
                    terminal_bound)
        last_global_bound = self.last_global_bound
        self._check_termination()
        if (self.queue.size() == 0) and \
           (self.termination_condition is None):
            self.termination_condition = \
                TerminationCondition.queue_empty
        if self.termination_condition is None:
            node = self._get_work_item()
            self.active_nodes = 1
            self.external_bound = node.bound
            if self.journalist is not None:
                force = (last_global_bound == \
                         self.converger.unbounded_objective) and \
                         (last_global_bound != \
                          self.last_global_bound)
                self.journalist.tic(force=force)
            return (False,
                    self.best_objective,
                    self.best_node,
                    node)
        else:
            if self.journalist is not None:
                self.journalist.tic(force=True)
                self.journalist.log_info(self.journalist._lines)
            self.initialized = False
            return (True,
                    self.best_objective,
                    self.best_node,
                    (self._get_current_bound(),
                     self.termination_condition,
                     self._get_final_solve_info()))

class DispatcherDistributed(DispatcherBase):
    """The central dispatcher for a distributed
    branch-and-bound algorithm.

    Parameters
    ----------
    comm : ``mpi4py.MPI.Comm``, optional
        The MPI communicator to use. If set to None, this
        will disable the use of MPI and avoid an attempted
        import of `mpi4py.MPI` (which avoids triggering a
        call to `MPI_Init()`).
    """

    def __init__(self, comm):
        assert comm.size > 1
        import mpi4py.MPI
        assert mpi4py.MPI.Is_initialized()
        super(DispatcherDistributed, self).__init__()
        self.clock = mpi4py.MPI.Wtime
        self.comm = comm
        # send rank of dispatcher to all workers
        self.dispatcher_rank = DispatcherProxy._init(
            self.comm,
            ProcessType.dispatcher)
        assert self.dispatcher_rank == self.comm.rank
        self.worker_ranks = [i for i in range(self.comm.size)
                             if i != self.comm.rank]
        self.needs_work_queue = \
            collections.deque([],
                              len(self.worker_ranks))
        self._solve_info_by_source = \
            {i: _SolveInfo() for i in self.worker_ranks}
        self.last_known_bound = dict()
        self.external_bounds = SortedList()
        self.has_work = set()
        self._send_requests = None
        self.explored_nodes_count = 0

    def _compute_load_imbalance(self):
        pmin = inf
        pmax = -inf
        psum = 0
        for solve_info in self._solve_info_by_source.values():
            count = solve_info.explored_nodes_count
            if count < pmin:
                pmin = count
            if count > pmax:
                pmax = count
            psum += count
        if psum > 0:
            pavg = psum/float(len(self._solve_info_by_source))
            return (pmax-pmin)/pavg*100.0
        else:
            return 0.0

    def _get_current_bound(self):
        """Get the current global bound"""
        bound = self.queue.bound()
        if self.converger.sense == maximize:
            if len(self.external_bounds) and \
               ((bound is None) or \
                (self.external_bounds[-1] > bound)):
                bound = self.external_bounds[-1]
            if (self.worst_terminal_bound is not None) and \
               ((bound is None) or \
                (self.worst_terminal_bound > bound)):
                bound = self.worst_terminal_bound
        else:
            if len(self.external_bounds) and \
               ((bound is None) or \
                (self.external_bounds[0] < bound)):
                bound = self.external_bounds[0]
            if (self.worst_terminal_bound is not None) and \
               ((bound is None) or \
                (self.worst_terminal_bound < bound)):
                bound = self.worst_terminal_bound
        if bound is not None:
            return bound
        else:
            # likely means the dispatcher was initialized
            # with an empty queue
            return self.converger.unbounded_objective

    def _get_final_solve_info(self):
        solve_info = _SolveInfo()
        for worker_solve_info in self._solve_info_by_source.values():
            solve_info.add_from(worker_solve_info)
        return solve_info

    def _get_node_counts(self):
        return (self.served_nodes_count,
                self.explored_nodes_count,
                self.queue.size() + len(self.has_work))

    #
    # Overloaded base class methods
    #

    def _check_update_best_objective(self, objective):
        updated = super(DispatcherDistributed, self).\
            _check_update_best_objective(objective)
        if updated:
            assert self.best_objective == objective
            self_external_bounds = self.external_bounds
            eligible_for_queue = self.converger.eligible_for_queue
            # trim the sorted external_bounds list
            N = len(self_external_bounds)
            if self.converger.sense == maximize:
                i = 0
                for i in range(N):
                    if eligible_for_queue(
                            self_external_bounds[i],
                            objective):
                        break
                if i != 0:
                    self.external_bounds = SortedList(
                        self_external_bounds.islice(i, N))
            else:
                i = N-1
                for i in range(N-1,-1,-1):
                    if eligible_for_queue(
                            self_external_bounds[i],
                            objective):
                        break
                if i != N-1:
                    self.external_bounds = SortedList(
                        self_external_bounds.islice(0, i+1))

    def _get_work_to_send(self, dest):
        node = self._get_work_item()
        bound = node.bound
        self.last_known_bound[dest] = bound
        self.external_bounds.add(bound)
        self.has_work.add(dest)
        return node

    def _send_work(self):
        stop = False
        data = None
        if len(self.needs_work_queue) > 0:
            if self._send_requests is None:
                self._send_requests = \
                    {i: None for i in self.worker_ranks}
            if self.termination_condition is None:
                while (self.queue.size() > 0) and \
                      (len(self.needs_work_queue) > 0):
                    stop = False
                    dest = self.needs_work_queue.popleft()
                    node = self._get_work_to_send(dest)
                    best_node_slots = None
                    if self.best_node is not None:
                        best_node_slots = self.best_node.slots
                    send_ = marshal.dumps(
                        (self.best_objective,
                         best_node_slots,
                         node.slots),
                        config.MARSHAL_PROTOCOL_VERSION)
                    if self._send_requests[dest] is not None:
                        self._send_requests[dest].Wait()
                    self._send_requests[dest] = \
                        self.comm.Isend([send_,mpi4py.MPI.BYTE],
                                        dest,
                                        tag=DispatcherResponse.work)
                    # a shortcut to check if we should keep sending nodes
                    if (self.node_limit is not None) and \
                       (self.served_nodes_count >= self.node_limit):
                        break
            if len(self.needs_work_queue) == (self.comm.size-1):
                if self.termination_condition is None:
                    self.termination_condition = \
                        TerminationCondition.queue_empty
                requests = []
                for r_ in self._send_requests.values():
                    if r_ is not None:
                        requests.append(r_)
                mpi4py.MPI.Request.Waitall(requests)
                self._send_requests = None
                stop = True
                data = (self._get_current_bound(),
                        self.termination_condition,
                        self._get_final_solve_info())
                best_node_slots = None
                if self.best_node is not None:
                    best_node_slots = self.best_node.slots
                send_ = marshal.dumps(
                    (self.best_objective,
                     best_node_slots,
                     data[0],
                     _termination_condition_to_int[data[1]],
                     data[2].data),
                    config.MARSHAL_PROTOCOL_VERSION)
                # everyone needs work, so we must be done
                requests = []
                while len(self.needs_work_queue) > 0:
                    dest = self.needs_work_queue.popleft()
                    requests.append(
                        self.comm.Isend([send_,mpi4py.MPI.BYTE],
                                        dest,
                                        DispatcherResponse.nowork))
                mpi4py.MPI.Request.Waitall(requests)

        return (stop, self.best_objective, self.best_node, data)

    def _update_solve_info(self, solve_info_data, source):
        self.explored_nodes_count -= \
            self._solve_info_by_source[source].explored_nodes_count
        self._solve_info_by_source[source].data[:] = solve_info_data
        self.explored_nodes_count += \
            self._solve_info_by_source[source].explored_nodes_count

    #
    # Interface
    #

    def initialize(self,
                   best_objective,
                   best_node,
                   initialize_queue,
                   queue_strategy,
                   converger,
                   node_limit,
                   time_limit,
                   queue_limit,
                   track_bound,
                   log,
                   log_interval_seconds,
                   log_new_incumbent):
        """Initialize the dispatcher. See the
        :func:`pybnb.dispatcher.DispatcherBase.initialize`
        method for argument descriptions."""
        self.needs_work_queue.clear()
        for solve_info in self._solve_info_by_source.values():
            solve_info.reset()
        self.last_known_bound.clear()
        self.external_bounds.clear()
        self.has_work.clear()
        self._send_requests = None
        self.explored_nodes_count = 0
        if best_node is not None:
            best_node = _SerializedNode.from_node(best_node)
        initialize_queue_ = DispatcherQueueData(
            nodes=[_SerializedNode.from_node(node) \
                   if (type(node) is not _SerializedNode) else node
                   for node in initialize_queue.nodes],
            worst_terminal_bound=initialize_queue.worst_terminal_bound,
            sense=initialize_queue.sense)
        super(DispatcherDistributed, self).initialize(
            best_objective,
            best_node,
            initialize_queue_,
            queue_strategy,
            converger,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        if self.journalist is not None:
            self.log_info("Starting branch & bound solve:\n"
                          " - dispatcher pid: %s (%s)\n"
                          " - worker processes: %d"
                          % (os.getpid(),
                             socket.gethostname(),
                             len(self.worker_ranks)))
            self.journalist.tic()

    def update(self,
               best_objective,
               best_node,
               terminal_bound,
               solve_info,
               node_list,
               source):
        """Update local worker information.

        Parameters
        ----------
        best_objective : float or None
            A new potential best objective found by the
            worker.
        best_node : :class:`Node <pybnb.node.Node>` or None
            A new potential best node found by the worker.
        terminal_bound : float or None
            The worst bound of any terminal nodes that were
            processed by the worker since the last update.
        solve_info : :class:`_SolveInfo`
            The most up-to-date worker solve information.
        node_list : list
            A list of nodes to add to the queue.
        source : int
            The worker process rank that the update came from.

        Returns
        -------
        solve_finished : bool
            Indicates if the dispatcher has terminated the solve.
        new_objective : float
            The best objective value known to the dispatcher.
        best_node : :class:`Node <pybnb.node.Node>` or None
            The best node known to the dispatcher.
        data : ``array.array`` or None
            If solve_finished is false, a data array
            representing a new node for the worker to
            process. Otherwise, a tuple containing the
            global bound, the termination condition string,
            and the number of explored nodes.
        """
        assert self.initialized
        self._update_solve_info(
            solve_info.data,
            source)
        self.needs_work_queue.append(source)
        self.has_work.discard(source)
        if source in self.last_known_bound:
            val_ = self.last_known_bound[source]
            try:
                self.external_bounds.remove(val_)
            except ValueError:
                # rare, but can happen when
                # _check_update_best_node modifies
                # the external_bounds list
                pass
        if best_objective is not None:
            self._check_update_best_objective(
                best_objective)
        if best_node is not None:
            assert best_node._uuid is not None
            self._check_update_best_node(best_node)
        if len(node_list) > 0:
            for node in node_list:
                self._add_work_to_queue(node)
        if terminal_bound is not None:
            self._check_update_worst_terminal_bound(
                terminal_bound)
        last_global_bound = self.last_global_bound
        self._check_termination()
        ret = self._send_work()
        stop = ret[0]
        if not stop:
            if self.journalist is not None:
                force = (last_global_bound == \
                         self.converger.unbounded_objective) and \
                         (last_global_bound != \
                          self.last_global_bound)
                self.journalist.tic(force=force)
        else:
            if self.journalist is not None:
                self.journalist.tic(force=True)
                self.journalist.log_info(self.journalist._lines)
            assert self.initialized
            self.initialized = False
        return ret

    #
    # Distributed Interface
    #

    def serve(self):
        """Start listening for distributed branch-and-bound
        commands and map them to commands in the local
        dispatcher interface."""

        def rebuild_update_requests(size):
            update_requests = {}
            update_data = bytearray(size)
            for i in self.worker_ranks:
                update_requests[i] = self.comm.Recv_init(
                    update_data,
                    source=i,
                    tag=DispatcherAction.update)
            return update_requests, update_data

        update_requests = None
        solve_info_ = _SolveInfo()
        data = None
        msg = Message(self.comm)
        while (1):
            msg.probe()
            tag = msg.tag
            source = msg.source
            if tag == DispatcherAction.update:
                size = msg.status.Get_count(datatype=mpi4py.MPI.BYTE)
                if (data is None) or \
                   (len(data) < size):
                    update_requests, data = \
                        rebuild_update_requests(size)
                req = update_requests[msg.status.Get_source()]
                req.Start()
                req.Wait()
                if six.PY2:
                    data_ = str(data)
                else:
                    data_ = data
                (best_objective,
                 best_node,
                 terminal_bound,
                 solve_info_data,
                 node_list) = marshal.loads(data_)
                solve_info_.data = array.array('d',solve_info_data)
                if best_node is not None:
                    best_node = _SerializedNode(best_node)
                node_list = [_SerializedNode(state)
                             for state in node_list]
                ret = self.update(best_objective,
                                  best_node,
                                  terminal_bound,
                                  solve_info_,
                                  node_list,
                                  source)
                stop = ret[0]
                if stop:
                    best_node = ret[2]
                    if best_node is not None:
                        best_node = _SerializedNode.restore_node(
                            best_node.slots)
                    return (ret[1],     # best_objective
                            best_node,
                            ret[3][0],  # global_bound
                            ret[3][1],  # termination_condition
                            ret[3][2])  # global_solve_info
            elif tag == DispatcherAction.log_info:
                msg.recv(mpi4py.MPI.CHAR)
                self.log_info(msg.data)
            elif tag == DispatcherAction.log_warning:
                msg.recv(mpi4py.MPI.CHAR)
                self.log_warning(msg.data)
            elif tag == DispatcherAction.log_debug:
                msg.recv(mpi4py.MPI.CHAR)
                self.log_debug(msg.data)
            elif tag == DispatcherAction.log_error:
                msg.recv(mpi4py.MPI.CHAR)
                self.log_error(msg.data)
            elif tag == DispatcherAction.log_critical:
                msg.recv(mpi4py.MPI.CHAR)
                self.log_critical(msg.data)
            elif tag == DispatcherAction.stop_listen:
                msg.recv()
                assert msg.data is None
                return (None, None, None, None, None)
            else:                                 #pragma:nocover
                raise RuntimeError("Dispatcher received invalid "
                                   "message tag '%s' from rank '%s'"
                                   % (tag, source))

    def save_dispatcher_queue(self):
        """Saves the current dispatcher queue. The result can
        be used to re-initialize a solve.

        Returns
        -------
        queue_data : :class:`pybnb.dispatcher.DispatcherQueueData`
            An object storing information that can be used
            to re-initialize the dispatcher queue to its
            current state.
        """
        nodes = []
        for node in self.queue.items():
            node_ = node.restore_node(node.slots)
            nodes.append(node_)
        return DispatcherQueueData(
            nodes=nodes,
            worst_terminal_bound=self.worst_terminal_bound,
            sense=self.converger.sense)
