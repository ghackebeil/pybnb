import sys
import time
import array
import collections

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

from pybnb.misc import (infinity,
                          get_gap_labels)
from pybnb.dispatcher_proxy import (ProcessType,
                                      DispatcherAction,
                                      WorkerAction,
                                      DispatcherProxy)
from pybnb.mpi_utils import Message
from pybnb.problem import (minimize,
                             maximize,
                             GenericProblem,
                             ProblemNode)

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

# This class used to store arrays in the priority queue
class _ProtectCompare(object):
    """Protects an object from being used for
    comparison. Instances of this class will always compare
    equal to something else."""
    __slots__ = ("obj",)
    __hash__ = None
    def __init__(self, obj):
        self.obj = obj
    def __lt__(self, other): return False
    def __gt__(self, other): return False
    def __le__(self, other): return True
    def __ge__(self, other): return True
    def __eq__(self, other): return True

class TreeIdLabeler(object):
    def __init__(self):
        self._next_id = 0
    def __call__(self):
        id_ = self._next_id
        self._next_id += 1
        return id_

SavedDispatcherQueue = collections.namedtuple("SavedDispatcherQueue",
                                              ["states","tree_id_labeler"])

class StatusPrinter(object):

    def __init__(self,
                 dispatcher,
                 log,
                 log_interval_seconds=1.0):
        assert log_interval_seconds >= 0
        self._dispatcher = dispatcher
        self._log_interval_seconds = log_interval_seconds
        self._log = log

        percent_relative_gap_tol = 1e-6
        if self._dispatcher.generic_problem.relative_gap_tolerance != 0:
            percent_relative_gap_tol = 100.0 * \
                self._dispatcher.generic_problem.relative_gap_tolerance
        rgap_label_str, rgap_number_str = \
            get_gap_labels(percent_relative_gap_tol, key="rgap")

        absolute_gap_tol = 1e-8
        if self._dispatcher.generic_problem.absolute_gap_tolerance != 0:
            absolute_gap_tol = \
                self._dispatcher.generic_problem.absolute_gap_tolerance
        agap_label_str, agap_number_str = \
            get_gap_labels(absolute_gap_tol, key="agap", format='g')

        if dispatcher.comm is None:
            self._time = time.time
        else:
            self._time = mpi4py.MPI.Wtime
        self._initial_header_line = \
            ("         Nodes        |"
             "                   Objective Bounds                     |"
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
        self._log.info(msg)

    def log_warn(self, msg):
        self._log.warn(msg)

    def log_debug(self, msg):
        self._log.debug(msg)

    def log_error(self, msg):
        self._log.error(msg)

    def new_objective(self, report=True):
        self._new_objective = True
        self._report_new_objective = report

    def tick(self, force=False):
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
            self._dispatcher.queue.qsize() + \
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
        bound = self._dispatcher.get_current_bound()
        objective = self._dispatcher.best_objective
        agap = self._dispatcher.generic_problem.\
               compute_absolute_gap(bound, objective)
        rgap = self._dispatcher.generic_problem.\
               compute_relative_gap(bound, objective)
        rgap *= 100.0
        if (self._print_count % 5) == 0:
            if self._print_count == 0:
                self._log.info(self._initial_header_line)
            self._log.info(self._header_line)

        if (rgap != infinity) and \
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
        self.generic_problem = None
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
        self.tree_id_labeler = None
        self.stop_optimality = False
        self.stop_node_limit = False
        self.stop_time_limit = False
        self.stop_cutoff = False
        self.initialized = False
        self._start_time = None

        if self.comm is not None:
            # send rank of dispatcher to all workers
            self.dispatcher_rank, self.root_worker_comm_rank, _ = \
                DispatcherProxy._init(
                    self.comm,
                    ProcessType.dispatcher)
            assert self.dispatcher_rank == self.comm.rank

            # another broadcast used by the workers for which
            # the result is not needed here
            self.comm.bcast(None,root=self.root_worker_comm_rank)
            self.worker_ranks = [i for i in range(self.comm.size)
                                 if i != self.comm.rank]
        else:
            self.dispatcher_rank = 0
            self.root_worker_comm_rank = 0
            self.worker_ranks = [0]

    def save_dispatcher_queue(self):
        return SavedDispatcherQueue(
            states=tuple(data.obj for _,data in self.queue.queue),
            tree_id_labeler=self.tree_id_labeler)

    def get_termination_condition(self):
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

    def _get_queue_bound(self):
        bound = None
        if self.queue.qsize() > 0:
            bound, data = self.queue.queue[0]
            if self.generic_problem.sense == maximize:
                bound = -bound
            assert bound == ProblemNode._extract_bound(data.obj)
        return bound

    def get_current_bound(self):
        bounds = []
        qbound = self._get_queue_bound()
        if qbound is not None:
            bounds.append(qbound)
        bounds.extend(self.last_known_bound.values())
        if self.worst_terminal_bound is not None:
            bounds.append(self.worst_terminal_bound)
        if self.generic_problem.sense == maximize:
            return max(bounds)
        else:
            assert self.generic_problem.sense == minimize
            return min(bounds)

    def _get_work_to_send(self, dest):
        priority, data = self.queue.get_nowait()
        bound = ProblemNode._extract_bound(data.obj)
        if self.generic_problem.sense == maximize:
            assert bound == -priority
        else:
            assert bound == priority
        self.last_known_bound[dest] = bound
        ProblemNode._insert_best_objective(
            data.obj,
            self.best_objective)
        self.has_work.add(dest)
        self.sent_nodes_count += 1
        return data.obj

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
                    if not self.queue.empty():
                        data = self._get_work_to_send(_source)
                        assert ProblemNode._extract_best_objective(data) == \
                            self.best_objective
                        return (self.best_objective, data)
            else:
                requests = []
                if not (self.stop_optimality or \
                        self.stop_node_limit or \
                        self.stop_time_limit or \
                        self.stop_cutoff):
                    while (not self.queue.empty()) and \
                          (not self.needs_work_queue.empty()):
                        dest = self.needs_work_queue.get_nowait()
                        data = self._get_work_to_send(dest)
                        requests.append(self.comm.Isend([data,mpi4py.MPI.DOUBLE],
                                                        dest,
                                                        tag=WorkerAction.work))
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
                                                        WorkerAction.nowork))
                    mpi4py.MPI.Request.Waitall(requests)

        return (self.best_objective, None)

    def serve(self):
        if self.comm is None:
            raise ValueError("The dispatcher was not instantiated "
                             "with an MPI communicator.")
        self._listen()

    def _check_update_worst_terminal_bound(self, bound):
        if (self.worst_terminal_bound is None) or \
           self.generic_problem.bound_worsened(bound,
                                               self.worst_terminal_bound):
            self.worst_terminal_bound = bound

    def _check_update_best_objective(self, objective):
        if self.generic_problem.objective_improved(objective,
                                                   self.best_objective):
            self.journalist.new_objective(report=True)
            self.best_objective = objective
            # now attempt to trim down the queue to save memory
            if self.queue.qsize() > 0:
                for ndx, (priority, data) in enumerate(self.queue.queue):
                    bound = ProblemNode._extract_bound(data.obj)
                    if not self.generic_problem.objective_can_improve(
                            self.best_objective,
                            bound):
                        break
                else:
                    ndx = len(self.queue.queue)
                if ndx != len(self.queue.queue):
                    # be sure update the worst terminal bound with
                    # any nodes we are throwing away (very unlikely to happen)
                    for i in range(ndx, len(self.queue.queue)):
                        self._check_update_worst_terminal_bound(
                            ProblemNode._extract_bound(self.queue.queue[i][1].obj))
                    self.queue.queue = self.queue.queue[:ndx]

    def _check_convergence(self):
        # check if we are done
        if not (self.stop_optimality or \
                self.stop_node_limit or \
                self.stop_time_limit or \
                self.stop_cutoff):
            global_bound = self.get_current_bound()
            if (global_bound == self.generic_problem.infeasible_objective) or \
               self.generic_problem.objective_is_optimal(self.best_objective,
                                                         global_bound):
                self.stop_optimality = True
            elif self.generic_problem.cutoff_is_met(global_bound):
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

    def _add_to_queue(self, state):
        bound = priority = ProblemNode._extract_bound(state)
        if self.generic_problem.sense == maximize:
            priority = -priority
        if self.generic_problem.objective_can_improve(self.best_objective,
                                                      bound):
            self.queue.put((priority,_ProtectCompare(state)))
            return True, bound
        else:
            return False, bound

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
                        state_size = int(msg.data[pos])
                        pos += 1
                        node_list[i] = msg.data[pos:pos+state_size]
                        pos += state_size
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
                self.comm.bcast(best_bound, root=self.comm.rank)
            elif tag == DispatcherAction.barrier:
                msg.recv()
                assert msg.data is None
                self.barrier()
                self.comm.Barrier()
            else:
                assert tag == DispatcherAction.solve_finished
                msg.recv()
                assert msg.data is None
                self.solve_finished()
                break

    def initialize(self,
                   best_objective,
                   initialize_queue,
                   generic_problem,
                   node_limit,
                   time_limit,
                   log,
                   log_interval_seconds):
        assert not self.initialized
        assert (node_limit is None) or \
            ((node_limit > 0) and \
             (node_limit == int(node_limit)))
        assert (time_limit is None) or \
            (time_limit >= 0)
        self.queue = Queue.PriorityQueue()
        self.needs_work_queue = Queue.Queue()
        self.generic_problem = generic_problem
        self.best_objective = generic_problem.infeasible_objective
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
        self.tree_id_labeler = initialize_queue.tree_id_labeler
        self.stop_optimality = False
        self.stop_node_limit = False
        self.stop_time_limit = False
        self.stop_cutoff = False
        self.initialized = True
        self._start_time = time.time()
        self.journalist.log_info("Running branch & bound (worker count: %d)"
                                 % (len(self.worker_ranks)))
        for state in initialize_queue.states:
            assert ProblemNode._has_tree_id(state)
            self._add_to_queue(state)
        self._check_update_best_objective(best_objective)
        self.journalist.tick()

    def update(self,
               best_objective,
               previous_bound,
               source_explored_nodes_count,
               node_states,
               _source=0):
        assert self.initialized
        self._update_explored_nodes_count(
            source_explored_nodes_count,
            _source)
        self.needs_work_queue.put(_source)
        self.has_work.discard(_source)
        if _source in self.last_known_bound:
            del self.last_known_bound[_source]
        self._check_update_best_objective(best_objective)
        if len(node_states):
            for state in node_states:
                if not ProblemNode._has_tree_id(state):
                    ProblemNode._insert_tree_id(state,
                                                self.tree_id_labeler())
                added, bound_ = self._add_to_queue(state)
                if not added:
                    self._check_update_worst_terminal_bound(bound_)
        else:
            if not self.first_update[_source]:
                self._check_update_worst_terminal_bound(previous_bound)
        self.first_update[_source] = False
        self._check_convergence()
        ret = self._send_work()
        self.journalist.tick()
        return ret

    def finalize(self):
        self.journalist.tick(force=True)
        best_bound = self.get_current_bound()
        assert self.initialized
        self.initialized = False
        return best_bound

    def barrier(self):
        # this is a no-op
        pass

    def solve_finished(self):
        # this is a no-op
        pass

    def log_info(self, msg):
        self.journalist.log_info(msg)

    def log_warning(self, msg):
        self.journalist.log_warn(msg)

    def log_debug(self, msg):
        self.journalist.log_debug(msg)

    def log_error(self, msg):
        self.journalist.log_error(msg)
