"""
Branch-and-bound dispatcher implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import array
import sys
import time
import array
import collections

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

from pybnb.common import (minimize,
                          maximize,
                          inf,
                          nan)
from pybnb.misc import get_gap_labels
from pybnb.dispatcher_proxy import (ProcessType,
                                    DispatcherAction,
                                    DispatcherResponse,
                                    DispatcherProxy)
from pybnb.node import Node
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.mpi_utils import (Message,
                             recv_nothing)
from pybnb.priority_queue import (WorstBoundFirstPriorityQueue,
                                  CustomPriorityQueue,
                                  BreadthFirstPriorityQueue,
                                  DepthFirstPriorityQueue)

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

class DispatcherQueueData(
        collections.namedtuple("DispatcherQueueData",
                               ["nodes","next_tree_id"])):
    """A namedtuple storing data that can be used
    re-initialize a dispatcher queue.

    Attributes
    ----------
    nodes : tuple
        A list of nodes stored in the order they were
        found in the priority queue.
    next_tree_id : int
        The next tree_id that will be assigned to a node.
    """

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
        if self._dispatcher.converger.relative_gap_tolerance != 0:
            percent_relative_gap_tol = 100.0 * \
                self._dispatcher.converger.relative_gap_tolerance
        rgap_label_str, rgap_number_str = \
            get_gap_labels(percent_relative_gap_tol, key="rgap")

        absolute_gap_tol = 1e-8
        if self._dispatcher.converger.absolute_gap_tolerance != 0:
            absolute_gap_tol = \
                self._dispatcher.converger.absolute_gap_tolerance
        agap_label_str, agap_number_str = \
            get_gap_labels(absolute_gap_tol, key="agap", format='g')

        if dispatcher.comm is None:
            self._time = time.time
        else:
            self._time = mpi4py.MPI.Wtime
        self._initial_header_line = \
            ("--------------------"
             "--------------------"
             "--------------------"
             "--------------------"
             "--------------------"
             "-------------\n"
             "         Nodes        |"
             "                     Objective Bounds                      |"
             "              Work              ")
        self._header_line = \
            (" {explored:>9} {unexplored:>9}  |{objective:>15} "
             "{bound:>15} "+rgap_label_str+"  "+agap_label_str+" |{runtime:>10} {rate:>10} "
             "{starved:>8}").\
             format(explored="Expl",
                    unexplored="Unexpl",
                    objective="Incumbent",
                    bound="Bound",
                    runtime="Time (s)",
                    rgap="Rel. Gap",
                    agap="Abs. Gap",
                    starved="Starved",
                    rate="Nodes/Sec")
        self._line_template = \
            ("{tag:>1}{explored:>9d} {unexplored:>9d}  |{objective:>15.7g} "
             "{bound:>15.7g} "+rgap_number_str+"% "+agap_number_str+" |{runtime:>10.2f} {rate:>10.2f} "
             "{starved:>8d}")
        self._line_template_big_gap = \
            ("{tag:>1}{explored:>9d} {unexplored:>9d}  |{objective:>15.7g} "
             "{bound:>15.7g} "+rgap_label_str+"% "+agap_number_str+" |{runtime:>10.2f} {rate:>10.2f} "
             "{starved:>8d}")

        self._start_time = self._time()
        self._last_print_time = float('-inf')
        assert self._dispatcher.sent_nodes_count == 0
        self._last_explored_nodes_count = 0
        self._last_rgap = None
        self._smoothing = 0.7
        self._avg_time_per_node = None
        self._print_count = 0
        self._new_objective = False
        self._report_new_objective = False

    def log_info(self, msg):
        """Pass a message to ``log.info``"""
        self._log.info(msg)

    def log_warn(self, msg):
        """Pass a message to ``log.warn``"""
        self._log.warn(msg)

    def log_debug(self, msg):
        """Pass a message to ``log.debug``"""
        self._log.debug(msg)

    def log_error(self, msg):
        """Pass a message to ``log.error``"""
        self._log.error(msg)

    def new_objective(self, report=True):
        """Indicate that a new objective has been found

        Parameters
        ----------
        report : bool, optional
            Indicate whether or not to force the next `tick`
            log output. (default: False)
        """
        self._new_objective = True
        self._report_new_objective = report

    def tick(self, force=False):
        """Provide an opportunity to log output if certain
        criteria are met.

        Parameters
        ----------
        force : bool, optional
            Indicate whether or not to force logging of
            output, even if logging criteria are not
            met. (default: False)
        """
        if self._log.disabled:
            return
        current_time = self._time()
        new_objective = self._new_objective
        report_new_objective = self._report_new_objective
        delta_t = current_time - self._last_print_time
        self._report_new_objective = False
        if not force:
            if not report_new_objective:
                if delta_t < self._log_interval_seconds:
                    return
        self._new_objective = False
        unexplored_nodes_count = \
            self._dispatcher.queue.size() + \
            len(self._dispatcher.has_work)
        explored_nodes_count = \
            self._dispatcher.explored_nodes_count
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
                runtime=current_time-self._start_time,
                starved=self._dispatcher.needs_work_queue.qsize(),
                rate=rate))
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
                runtime=current_time-self._start_time,
                starved=self._dispatcher.needs_work_queue.qsize(),
                rate=rate))
            self._last_rgap = rgap
        self._last_explored_nodes_count = explored_nodes_count
        self._print_count += 1
        self._last_print_time = current_time

class Dispatcher(object):
    """The central dispatcher for a distributed
    branch-and-bound algorithm.

    Parameters
    ----------
    comm : ``mpi4py.MPI.Comm``, optional
        The MPI communicator to use. If set to None, this
        will disable the use of MPI and avoid an attempted
        import of mpi4py.MPI (which avoids triggering a call
        to `MPI_Init()`).
    """

    def __init__(self, comm):
        if comm is not None:
            import mpi4py.MPI
            assert mpi4py.MPI.Is_initialized()
        self.comm = comm
        self.worker_ranks = []
        # the following attributes will be reset each
        # time initialize is called
        self.queue = None
        self.needs_work_queue = None
        self.converger = None
        self.journalist = None
        self.best_objective = None
        self.node_limit = None
        self.time_limit = None
        self.sent_nodes_count = None
        self._explored_nodes_count_by_source = None
        self.explored_nodes_count = 0
        self.last_known_bound = None
        self.worst_terminal_bound = None
        self.first_update = None
        self.has_work = None
        self.next_tree_id = None
        self.stop_optimality = False
        self.stop_node_limit = False
        self.stop_time_limit = False
        self.stop_cutoff = False
        self.initialized = False
        self._start_time = None

        if self.comm is not None:
            # send rank of dispatcher to all workers
            self.dispatcher_rank, self.root_worker_rank = \
                DispatcherProxy._init(
                    self.comm,
                    ProcessType.dispatcher)
            assert self.dispatcher_rank == self.comm.rank
            self.worker_ranks = [i for i in range(self.comm.size)
                                 if i != self.comm.rank]
        else:
            self.dispatcher_rank = 0
            self.worker_ranks = [0]

    def _add_work_to_queue(self, data):
        bound = Node._extract_bound(data)
        if self.converger.objective_can_improve(
                self.best_objective,
                bound):
            self.queue.put(data)
            return True
        else:
            self._check_update_worst_terminal_bound(
                Node._extract_bound(data))
            return False

    def _get_current_bound(self):
        bounds = []
        if self.queue.size() > 0:
            bounds.append(self.queue.bound())
        bounds.extend(self.last_known_bound.values())
        if self.worst_terminal_bound is not None:
            bounds.append(self.worst_terminal_bound)
        if self.converger.sense == maximize:
            return max(bounds)
        else:
            assert self.converger.sense == minimize
            return min(bounds)

    def _get_work_to_send(self, dest):
        data = self.queue.get()
        assert data is not None
        bound = Node._extract_bound(data)
        self.last_known_bound[dest] = bound
        Node._insert_best_objective(
            data,
            self.best_objective)
        self.has_work.add(dest)
        self.sent_nodes_count += 1
        return data

    def _send_work(self):

        if not self.needs_work_queue.empty():
            if self.comm is None:
                assert self.needs_work_queue.qsize() == 1
                _source = self.needs_work_queue.get_nowait()
                assert _source == 0
                if not (self.stop_optimality or \
                        self.stop_node_limit or \
                        self.stop_time_limit or \
                        self.stop_cutoff):
                    if self.queue.size() > 0:
                        data = self._get_work_to_send(_source)
                        assert Node._extract_best_objective(data) == \
                            self.best_objective
                        return (self.best_objective, data)
            else:
                requests = []
                if not (self.stop_optimality or \
                        self.stop_node_limit or \
                        self.stop_time_limit or \
                        self.stop_cutoff):
                    while (self.queue.size() > 0) and \
                          (not self.needs_work_queue.empty()):
                        dest = self.needs_work_queue.get_nowait()
                        data = self._get_work_to_send(dest)
                        requests.append(self.comm.Isend([data,mpi4py.MPI.DOUBLE],
                                                        dest,
                                                        tag=DispatcherResponse.work))
                if len(requests):
                    mpi4py.MPI.Request.Waitall(requests)
                elif self.needs_work_queue.qsize() == (self.comm.size-1):
                    assert len(requests) == 0
                    send_data = array.array('d',[self.best_objective])
                    # everyone needs work, so we must be done
                    while not self.needs_work_queue.empty():
                        dest = self.needs_work_queue.get()
                        requests.append(self.comm.Isend([send_data,mpi4py.MPI.DOUBLE],
                                                        dest,
                                                        DispatcherResponse.nowork))
                    mpi4py.MPI.Request.Waitall(requests)

        return (self.best_objective, None)

    def _check_update_worst_terminal_bound(self, bound):
        if (self.worst_terminal_bound is None) or \
           self.converger.bound_worsened(bound,
                                         self.worst_terminal_bound):
            self.worst_terminal_bound = bound

    def _check_update_best_objective(self, objective):
        if self.converger.objective_improved(objective,
                                             self.best_objective):
            self.journalist.new_objective(report=True)
            self.best_objective = objective
            removed = self.queue.filter(
                lambda data_: self.converger.objective_can_improve(
                    objective,
                    Node._extract_bound(data_)))
            for data in removed:
                self._check_update_worst_terminal_bound(
                    Node._extract_bound(data))

    def _check_convergence(self):
        # check if we are done
        if not (self.stop_optimality or \
                self.stop_node_limit or \
                self.stop_time_limit or \
                self.stop_cutoff):
            global_bound = self._get_current_bound()
            if (global_bound == self.converger.infeasible_objective) or \
               self.converger.objective_is_optimal(self.best_objective,
                                                   global_bound):
                self.stop_optimality = True
            elif self.converger.cutoff_is_met(global_bound):
                self.stop_cutoff = True
            if not (self.stop_optimality or \
                    self.stop_cutoff):
                if (self.node_limit is not None) and \
                   (self.explored_nodes_count >= self.node_limit):
                    self.stop_node_limit = True
                if not self.stop_node_limit:
                    if (self.time_limit is not None) and \
                       ((time.time() - self._start_time) >= self.time_limit):
                        self.stop_time_limit = True

    def _update_explored_nodes_count(self, explored_count, source):
        self.explored_nodes_count -= \
            self._explored_nodes_count_by_source[source]
        self.explored_nodes_count += explored_count
        self._explored_nodes_count_by_source[source] = \
            explored_count

    def _listen(self):
        msg = Message(self.comm)
        while (1):
            msg.probe()
            tag = msg.tag
            source = msg.source
            if tag == DispatcherAction.update:
                msg.recv(mpi4py.MPI.DOUBLE)
                best_objective = float(msg.data[0])
                previous_bound = float(msg.data[1])
                assert int(msg.data[2]) == msg.data[2]
                source_explored_nodes_count = int(msg.data[2])
                assert int(msg.data[3]) == msg.data[3]
                nodes_receiving_count = int(msg.data[3])
                if nodes_receiving_count > 0:
                    node_list = [None]*nodes_receiving_count
                    pos = 4
                    for i in range(nodes_receiving_count):
                        assert int(msg.data[pos]) == msg.data[pos]
                        data_size = int(msg.data[pos])
                        pos += 1
                        node_list[i] = msg.data[pos:pos+data_size]
                        pos += data_size
                else:
                    node_list = ()
                self.update(best_objective,
                            previous_bound,
                            source_explored_nodes_count,
                            node_list,
                            _source=source)
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
            elif tag == DispatcherAction.finalize:
                msg.recv()
                assert msg.data is None
                best_bound = self.finalize()
                assert best_bound is not None
                self.comm.Bcast(
                    [array.array("d",[best_bound]),
                     mpi4py.MPI.DOUBLE],
                    root=self.comm.rank)
                break
            elif tag == DispatcherAction.stop_listen:
                msg.recv()
                assert msg.data is None
                break
    #
    # Local Interface
    #

    def initialize(self,
                   best_objective,
                   initialize_queue,
                   node_priority_strategy,
                   converger,
                   node_limit,
                   time_limit,
                   log,
                   log_interval_seconds):
        """Initialize the dispatcher for a new solve.

        Parameters
        ----------
        best_objective : float
            The assumed best objective to start with.
        initialize_queue : :class:`pybnb.dispatcher.DispatcherQueueData`
            The initial queue.
        node_priority_strategy : {"bound", "breadth", "depth", "custom"}
            Indicates the strategy for ordering nodes in the
            work queue. The "bound" strategy always selects
            the node with the worst bound first. The
            "breadth" strategy always selects the node with
            the smallest tree depth first (i.e.,
            breadth-first search). The "depth" strategy
            always selects the node with the largest tree
            depth first (i.e., depth-first search). The
            "custom" strategy assumes the
            :attr:`queue_priority
            <pybnb.node.Node.queue_priority>` node attribute
            has been set by the user. For all other
            strategies, the :attr:`queue_priority
            <pybnb.node.Node.queue_priority>` node attribute
            will be set automatically. In all cases, the
            largest priority node is always selected first,
            with ties being broken by insertion order.
        converger : :class:`pybnb.convergence_checker.ConvergenceChecker`
            The branch-and-bound convergence checker object.
        node_limit : int or None
            In integer representing the maximum number of
            nodes to processes before beginning to terminate
            the solve. If None, no node limit will be enforced.
        time_limit : float
            The maximum amount of time to spend processing
            nodes before beginning to terminate the
            solve. If None, no time limit will be enforced.
        log : ``logging.Logger``
            A log object where solver output should be sent.
        log_interval_seconds : float
            The approximate maximum time (in seconds)
            between solver log updates. More time may pass
            between log updates if no updates have been
            received from any workers, and less time may
            pass if a new incumbent is found. (default: 1.0)
        """
        assert (node_limit is None) or \
            ((node_limit > 0) and \
             (node_limit == int(node_limit)))
        assert (time_limit is None) or \
            (time_limit >= 0)
        self.needs_work_queue = Queue.Queue()
        self.converger = converger
        self.best_objective = converger.infeasible_objective
        if node_priority_strategy == "bound":
            self.queue = WorstBoundFirstPriorityQueue(
                self.converger.sense)
        elif node_priority_strategy == "custom":
            self.queue = CustomPriorityQueue(
                self.converger.sense)
        elif node_priority_strategy == "breadth":
            self.queue = BreadthFirstPriorityQueue(
                self.converger.sense)
        else:
            assert node_priority_strategy == "depth"
            self.queue = DepthFirstPriorityQueue(
                self.converger.sense)
        self.node_limit = None
        if node_limit is not None:
            self.node_limit = int(node_limit)
        self.time_limit = None
        if time_limit is not None:
            self.time_limit = float(time_limit)

        self.sent_nodes_count = 0
        self._explored_nodes_count_by_source = \
            {i: 0 for i in self.worker_ranks}
        self.last_known_bound = dict()
        self.worst_terminal_bound = None
        self.explored_nodes_count = 0
        self.first_update = \
            {_r: True for _r in self.worker_ranks}
        self.has_work = set()
        self.journalist = StatusPrinter(
            self,
            log,
            log_interval_seconds=log_interval_seconds)
        self.next_tree_id = initialize_queue.next_tree_id
        self.stop_optimality = False
        self.stop_node_limit = False
        self.stop_time_limit = False
        self.stop_cutoff = False
        self.initialized = True
        self._start_time = time.time()
        self.journalist.log_info("Starting branch & bound solve:\n"
                                 " - worker processes: %d\n"
                                 " - node priority strategy: %s"
                                 % (len(self.worker_ranks),
                                    node_priority_strategy))
        for node in initialize_queue.nodes:
            assert node.tree_id is not None
            assert self.next_tree_id > node.tree_id
            added = self._add_work_to_queue(node._data)
            assert added
        self._check_update_best_objective(best_objective)
        self.journalist.tick()

    def update(self,
               best_objective,
               previous_bound,
               source_explored_nodes_count,
               node_data,
               _source=0):
        """Update local worker information.

        Parameters
        ----------
        best_objective : float
            The current best objective value known to the
            worker.
        previous_bound : float
            The updated bound computed for the last node
            that was processed by the worker.
        source_explored_nodes_count : int
            The total number of nodes explored by the
            worker.
        node_data : list
            A list of new node data arrays to add to the queue.

        Returns
        -------
        new_objective : float
            The best objective value known to the dispatcher.
        data : ``array.array`` or None
            A data array representing a new node for the
            worker to process. If None, this indicates that
            the worker should begin to finalize the solve.
        """
        assert self.initialized
        self._update_explored_nodes_count(
            source_explored_nodes_count,
            _source)
        self.needs_work_queue.put(_source)
        self.has_work.discard(_source)
        if _source in self.last_known_bound:
            del self.last_known_bound[_source]
        self._check_update_best_objective(best_objective)
        if len(node_data):
            for data in node_data:
                assert not Node._has_tree_id(data)
                Node._insert_tree_id(data,
                                     self.next_tree_id)
                self.next_tree_id += 1
                self._add_work_to_queue(data)
        else:
            if not self.first_update[_source]:
                self._check_update_worst_terminal_bound(
                    previous_bound)
        self.first_update[_source] = False
        self._check_convergence()
        ret = self._send_work()
        self.journalist.tick()
        return ret

    def finalize(self):
        """Start the solve finalization.

        Returns
        -------
        best_bound : float
            The best objective bound known to the
            dispatcher.
        """
        self.journalist.tick(force=True)
        self.journalist.log_info("--------------------"
                                 "--------------------"
                                 "--------------------"
                                 "--------------------"
                                 "--------------------"
                                 "-------------")
        best_bound = self._get_current_bound()
        assert self.initialized
        self.initialized = False
        return best_bound

    def log_info(self, msg):
        """Pass a message to ``log.info``"""
        self.journalist.log_info(msg)

    def log_warning(self, msg):
        """Pass a message to ``log.warn``"""
        self.journalist.log_warn(msg)

    def log_debug(self, msg):
        """Pass a message to ``log.debug``"""
        self.journalist.log_debug(msg)

    def log_error(self, msg):
        """Pass a message to ``log.error``"""
        self.journalist.log_error(msg)

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
            nodes=[Node(data_=data)
                   for data in self.queue.items()],
            next_tree_id=self.next_tree_id)

    def get_termination_condition(self):
        """Get the solve termination description.

        Returns
        -------
        termination_condition : string
            A string describing the reason for solve termination.
        """
        if self.stop_optimality:
            return "optimality"
        elif self.stop_node_limit:
            return "node_limit"
        elif self.stop_time_limit:
            return "time_limit"
        elif self.stop_cutoff:
            return "cutoff"
        else:
            return "no_nodes"

    #
    # Distributed Interface
    #

    def serve(self):
        """Start listening for distributed branch-and-bound
        commands and map them to commands in the local
        dispatcher interface."""
        if self.comm is None:
            raise ValueError("The dispatcher was not instantiated "
                             "with an MPI communicator.")
        self._listen()
