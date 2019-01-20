"""
Branch-and-bound solver implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import sys
import time

from pybnb.common import (minimize,
                          maximize,
                          QueueStrategy,
                          TerminationCondition,
                          SolutionStatus)
from pybnb.problem import (_SolveInfo,
                           _SimpleSolveInfoCollector,
                           _ProblemWithSolveInfoCollection)
from pybnb.misc import (MPI_InterruptHandler,
                        time_format,
                        as_stream,
                        get_simple_logger)
from pybnb.node import Node
from pybnb.convergence_checker import (_default_scale,
                                       ConvergenceChecker)
from pybnb.dispatcher_proxy import DispatcherProxy
from pybnb.dispatcher import (DispatcherLocal,
                              DispatcherDistributed,
                              DispatcherQueueData)

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

import six

class _notset(object):
    pass

class SolverResults(object):
    """Stores the results of a branch-and-bound solve.

    Attributes
    ----------
    solution_status : :class:`SolutionStatus <pybnb.common.SolutionStatus>`
        The solution status. This attribute is comparable
        with strings as well as attributes of the
        :class:`SolutionStatus <pybnb.common.SolutionStatus>`
        enum.

        Example
        -------

        >>> import pybnb
        >>> results = pybnb.SolverResults()
        >>> results.solution_status = pybnb.SolutionStatus.optimal
        >>> assert results.solution_status == "optimal"
        >>> assert results.solution_status == pybnb.SolutionStatus.optimal
        >>> assert results.solution_status.value == "optimal"

    termination_condition : :class:`TerminationCondition <pybnb.common.TerminationCondition>`
        The solve termination condition, as
        determined by the dispatcher. This attribute is comparable
        with strings as well as attributes of the
        :class:`TerminationCondition <pybnb.common.TerminationCondition>`
        enum.

        Example
        -------

        >>> import pybnb
        >>> results = pybnb.SolverResults()
        >>> results.termination_condition = pybnb.TerminationCondition.optimality
        >>> assert results.termination_condition == "optimality"
        >>> assert results.termination_condition == pybnb.TerminationCondition.optimality
        >>> assert results.termination_condition.value == "optimality"

    objective : float
        The best objective found.
    bound : float
        The global optimality bound.
    absolute_gap : float
        The absolute gap between the objective and bound.
    relative_gap : float
        The relative gap between the objective and bound.
    nodes : float
        The total number of nodes processes by all workers.
    wall_time : float
        The process-local wall time (seconds). This is the
        only value on the results object that varies between
        processes.
    """

    def __init__(self):
        self.solution_status = None
        self.termination_condition = None
        self.objective = None
        self.bound = None
        self.absolute_gap = None
        self.relative_gap = None
        self.nodes = None
        self.wall_time = None

    def pprint(self, stream=sys.stdout):
        """Prints a nicely formatted representation of the
        results.

        Parameters
        ----------
        stream : file-like object or string, optional
            A file-like object or a filename where results
            should be written to. (default: ``sys.stdout``)
        """
        with as_stream(stream) as stream:
            stream.write("solver results:\n")
            self.write(stream, prefix=" - ", pretty=True)

    def write(self, stream, prefix="", pretty=False):
        """Writes results in YAML format to a stream or
        file.

        Parameters
        ----------
        stream : file-like object or string
            A file-like object or a filename where results
            should be written to.
        prefix : string, optional
            A string to use as a prefix for each line that
            is written. (default: '')
        pretty : bool, optional
            Indicates whether or not certain recognized
            attributes should be formatted for more
            human-readable output. (default: False)
        """
        with as_stream(stream) as stream:
            attrs = vars(self)
            names = sorted(list(attrs.keys()))
            first = ('solution_status', 'termination_condition',
                     'objective', 'bound',
                     'absolute_gap', 'relative_gap',
                     'nodes', 'wall_time')
            for cnt, name in enumerate(first):
                if not hasattr(self, name):
                    continue
                names.remove(name)
                val = getattr(self, name)
                if val is not None:
                    if pretty:
                        if name == 'wall_time':
                            val = time_format(val, digits=2)
                        elif name in ('objective','bound',
                                      'absolute_gap','relative_gap'):
                            val = "%.7g" % (val)
                    if name in ("solution_status", "termination_condition"):
                        val = val.value
                stream.write(prefix+'%s: %s\n'
                             % (name, val))
            for name in names:
                stream.write(prefix+'%s: %s\n'
                              % (name, getattr(self, name)))

    def __str__(self):
        """Represents the results as a string."""
        tmp = six.StringIO()
        self.pprint(stream=tmp)
        return tmp.getvalue()

class Solver(object):
    """A branch-and-bound solver.

    Parameters
    ----------
    comm : ``mpi4py.MPI.Comm``, optional
        The MPI communicator to use. If unset, the
        mpi4py.MPI.COMM_WORLD communicator will be
        used. Setting this keyword to None will disable the
        use of MPI and avoid an attempted import of
        mpi4py.MPI (which avoids triggering a call to
        `MPI_Init()`).
    dispatcher_rank : int, optional
        The process with this rank will be designated as the
        dispatcher process. If MPI functionality is disabled
        (by setting comm=None), this keyword must be 0.
        (default: 0)
    """

    def __init__(self,
                 comm=_notset,
                 dispatcher_rank=0):
        mpi = True
        if comm is None:
            mpi = False
        self._comm = None
        self._worker_flag = None
        self._dispatcher_flag = None
        self._disp = None
        self._time = None
        if mpi:
            import mpi4py.MPI
            assert mpi4py.MPI.Is_initialized()
            assert comm is not None
            if comm is _notset:
                comm = mpi4py.MPI.COMM_WORLD
            if (int(dispatcher_rank) != dispatcher_rank) or \
               (dispatcher_rank < 0) or \
               (dispatcher_rank >= comm.size):
                raise ValueError("The 'dispatcher_rank' keyword "
                                 "has been set to %s, which is not "
                                 "an available rank given the "
                                 "size of the MPI communicator (%d)."
                                 % (dispatcher_rank, comm.size))
            self._comm = comm
            if comm.size > 1:
                dispatcher_rank = int(dispatcher_rank)
                if comm.rank == dispatcher_rank:
                    self._disp = DispatcherDistributed(comm)
                    self._worker_flag = False
                    self._dispatcher_flag = True
                else:
                    self._disp = DispatcherProxy(comm)
                    self._worker_flag = True
                    self._dispatcher_flag = False
            else:
                self._disp = DispatcherLocal()
                self._worker_flag = True
                self._dispatcher_flag = True
            self._time = mpi4py.MPI.Wtime
        else:
            if dispatcher_rank != 0:
                raise ValueError(
                    "MPI functionality has been disabled but "
                    "the 'dispatcher_rank' keyword is set to "
                    "something other than 0.")
            assert self._comm is None
            self._disp = DispatcherLocal()
            self._worker_flag = True
            self._dispatcher_flag = True
            self._time = time.time
        assert self._worker_flag in (True, False)
        assert self._dispatcher_flag in (True, False)
        assert self._disp is not None
        assert self._time is not None
        self._solve_start = None
        self._wall_time = 0.0
        self._best_objective = None
        self._local_solve_info = _SolveInfo()
        self._global_solve_info = None

    def _reset_local_solve_stats(self):
        self._solve_start = None
        self._wall_time = 0.0
        self._best_objective = None
        self._local_solve_info.reset()
        self._global_solve_info = None

    def _check_update_best_objective(self,
                                     convergence_checker,
                                     new_objective):
        if convergence_checker.objective_improved(
                new_objective,
                self._best_objective):
            self._best_objective = new_objective
            return True
        else:
            return False

    def _fill_results(self, results, convergence_checker):
        if results.bound == convergence_checker.infeasible_objective:
            assert results.objective == \
                convergence_checker.infeasible_objective, \
                str(results.objective)
            results.solution_status = SolutionStatus.infeasible
        elif results.objective == \
             convergence_checker.infeasible_objective:
            results.solution_status = SolutionStatus.unknown
        elif results.objective == \
             convergence_checker.unbounded_objective:
            assert results.bound == \
                convergence_checker.unbounded_objective, \
                str(results.bound)
            results.solution_status = SolutionStatus.unbounded
        else:
            results.absolute_gap = convergence_checker.\
                                   compute_absolute_gap(results.bound,
                                                        results.objective)
            results.relative_gap = convergence_checker.\
                                   compute_relative_gap(results.bound,
                                                        results.objective)
            if convergence_checker.objective_is_optimal(
                    results.objective,
                    results.bound):
                results.solution_status = SolutionStatus.invalid
                if (convergence_checker.sense == minimize) and \
                   (results.bound <= results.objective):
                    results.solution_status = SolutionStatus.optimal
                elif (convergence_checker.sense == maximize) and \
                     (results.bound >= results.objective):
                    results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible

    def _solve(self,
               problem,
               best_objective,
               disable_objective_call,
               convergence_checker,
               results):
        infeasible_objective = problem.infeasible_objective()
        assert infeasible_objective == \
            convergence_checker.infeasible_objective
        unbounded_objective = problem.unbounded_objective()
        assert unbounded_objective == \
            convergence_checker.unbounded_objective

        self._best_objective = best_objective
        children = ()
        bound = unbounded_objective
        if not isinstance(problem, _ProblemWithSolveInfoCollection):
            problem = _SimpleSolveInfoCollector(problem)
            problem.set_clock(self._time)
        problem.set_solve_info_object(self._local_solve_info)

        working_node = Node()
        assert working_node.tree_id is None
        # start the work loop
        while (1):
            update_start = self._time()
            stop, new_objective, data = \
                self._disp.update(
                    self._best_objective,
                    bound,
                    self._local_solve_info,
                    children)
            update_stop = self._time()
            self._local_solve_info._increment_queue_stat(
                update_stop-update_start, 1)

            updated = self._check_update_best_objective(
                convergence_checker,
                new_objective)
            if updated and \
               (self._best_objective != unbounded_objective):
                problem.notify_new_best_objective_received(
                    self._best_objective)
            del updated

            children = []

            if stop:
                # make sure all processes have the exact same best
                # objective value (not just subject to tolerances)
                self._best_objective = new_objective
                break
            # load the new data into the working_node
            working_node._set_data(data)
            del new_objective
            del data
            self._local_solve_info._increment_explored_nodes_stat(1)

            bound = working_node.bound
            current_tree_id = working_node.tree_id
            current_tree_depth = working_node.tree_depth
            assert current_tree_id is not None
            assert current_tree_depth >= 0

            # we should not be receiving a node that
            # does not satisfy these assertions
            assert convergence_checker.eligible_for_queue(
                    bound,
                    self._best_objective)

            problem.load_state(working_node)

            new_bound = problem.bound()
            if convergence_checker.bound_worsened(new_bound, bound):    #pragma:nocover
                self._disp.log_warning(
                    "WARNING: Bound became worse "
                    "(old=%r, new=%r)"
                    % (bound, new_bound))
            working_node.bound = new_bound
            bound = new_bound

            if convergence_checker.eligible_for_queue(
                    bound,
                    self._best_objective):
                objective = working_node.objective
                if not disable_objective_call:
                    objective = problem.objective()
                    working_node.objective = objective
                if convergence_checker.best_bound(bound, objective) != objective: #pragma:nocover
                    self._disp.log_warning(
                        "WARNING: Local node bound is worse "
                        "than local node objective (bound=%r, "
                        "objective=%r)" % (bound, objective))
                updated = self._check_update_best_objective(
                    convergence_checker,
                    objective)
                if updated and \
                   (self._best_objective != unbounded_objective):
                    problem.notify_new_best_objective(
                        self._best_objective)
                del updated
                if (objective != unbounded_objective) and \
                    convergence_checker.eligible_for_queue(
                        bound,
                        self._best_objective) and \
                    convergence_checker.eligible_to_branch(
                        bound,
                        objective):
                    clist = problem.branch(working_node)
                    for child in clist:
                        assert child.parent_tree_id == current_tree_id
                        assert child.tree_id is None
                        assert child.tree_depth >= current_tree_depth + 1
                        assert child.objective == working_node.objective
                        children.append(child._data)
                        if convergence_checker.bound_worsened(child.bound, bound):    #pragma:nocover
                            self._disp.log_warning(
                                "WARNING: Bound on child node "
                                "returned from branch method "
                                "is worse than parent node "
                                "(child=%r, parent=%r)"
                                % (child.bound, bound))

        assert len(data) == 3
        global_bound = data[0]
        termination_condition = data[1]
        global_solve_info = data[2]
        return (self._best_objective,
                global_bound,
                termination_condition,
                global_solve_info)

    #
    # Interface
    #

    @property
    def is_worker(self):
        """Indicates if this process has been designated as
        a worker."""
        return self._worker_flag

    @property
    def is_dispatcher(self):
        """Indicates if this process has been designated as
        the dispatcher."""
        return self._dispatcher_flag

    @property
    def comm(self):
        """The full MPI communicator that includes the
        dispatcher and all workers. Will be None if MPI
        functionality has been disabled."""
        return self._comm

    @property
    def worker_comm(self):
        """The worker MPI communicator. Will be None on any
        processes for which :attr:`Solver.is_worker` is
        False, or if MPI functionality has been disabled."""
        if (self._comm is None) or \
           (self._comm.size == 1):
            return self._comm
        elif not self.is_dispatcher:
            return self._disp.worker_comm
        return None

    @property
    def worker_count(self):
        """The number of worker processes associated with this solver."""
        if (self._comm is None) or \
           (self._comm.size == 1):
            return 1
        elif not self.is_dispatcher:
            return self._disp.worker_comm.size
        else:
            return len(self._disp.worker_ranks)

    def collect_worker_statistics(self):
        """Collect individual worker statistics about the
        most recent solve.

        Returns
        -------
        dict
            A dictionary whose keys are the different
            statistics collected, where each entry is a list
            storing a value for each worker.
        """
        import numpy
        stats = {}
        if (self.comm is not None) and \
           (self.comm.size > 1):
            gathered = numpy.empty((self.worker_count, 12),
                                   dtype=float)
            if self.is_worker:
                assert self.worker_comm is not None
                assert not self.is_dispatcher
                solve_info = self._local_solve_info
                mine = numpy.array(
                    [self._wall_time,
                     solve_info.total_queue_time,
                     solve_info.queue_call_count,
                     solve_info.total_objective_time,
                     solve_info.objective_call_count,
                     solve_info.total_bound_time,
                     solve_info.bound_call_count,
                     solve_info.total_branch_time,
                     solve_info.branch_call_count,
                     solve_info.total_load_state_time,
                     solve_info.load_state_call_count,
                     solve_info.explored_nodes_count],
                    dtype=float)
                assert len(mine) == gathered.shape[1]
                self.worker_comm.Allgather([mine, mpi4py.MPI.DOUBLE],
                                           [gathered, mpi4py.MPI.DOUBLE])
                if self.worker_comm.rank == 0:
                    self.comm.Send([gathered, mpi4py.MPI.DOUBLE],
                                   self._disp.dispatcher_rank,
                                   tag=11112111)
            else:
                assert self.worker_comm is None
                assert self.is_dispatcher
                self.comm.Recv([gathered, mpi4py.MPI.DOUBLE],
                               tag=11112111)
            gathered = gathered.T.tolist()
            stats['wall_time'] = gathered[0]
            stats['queue_time'] = gathered[1]
            stats['queue_call_count'] = gathered[2]
            stats['objective_time'] = gathered[3]
            stats['objective_call_count'] = gathered[4]
            stats['bound_time'] = gathered[5]
            stats['bound_call_count'] = gathered[6]
            stats['branch_time'] = gathered[7]
            stats['branch_call_count'] = gathered[8]
            stats['load_state_time'] = gathered[9]
            stats['load_state_call_count'] = gathered[10]
            stats['explored_nodes_count'] = gathered[11]
        else:
            assert self.is_worker
            assert self.is_dispatcher
            solve_info = self._local_solve_info
            stats['wall_time'] = [self._wall_time]
            stats['queue_time'] = [solve_info.total_queue_time]
            stats['queue_call_count'] = [solve_info.queue_call_count]
            stats['objective_time'] = \
                [solve_info.total_objective_time]
            stats['objective_call_count'] = \
                [solve_info.objective_call_count]
            stats['bound_time'] = \
                [solve_info.total_bound_time]
            stats['bound_call_count'] = \
                [solve_info.bound_call_count]
            stats['branch_time'] = \
                [solve_info.total_branch_time]
            stats['branch_call_count'] = \
                [solve_info.branch_call_count]
            stats['load_state_time'] = \
                [solve_info.total_load_state_time]
            stats['load_state_call_count'] = \
                [solve_info.load_state_call_count]
            stats['explored_nodes_count'] = \
                [solve_info.explored_nodes_count]

        return stats

    def save_dispatcher_queue(self):
        """Saves the dispatcher queue.

        Returns
        -------
        queue : :class:`pybnb.dispatcher.DispatcherQueueData` or None
            If this process is the dispatcher, this method
            will return an object storing any nodes
            currently in the dispatcher queue.  If this
            process is not the dispatcher, this method will
            return None.  The returned object can be used to
            reinitialize a solve (e.g., with different
            algorithms settings) by assigning it to the
            initialize_queue keyword of the
            :func:`Solver.solve` method.
        """
        ret = None
        if self.is_dispatcher:
            ret = self._disp.save_dispatcher_queue()
        return ret

    def solve(self,
              problem,
              best_objective=None,
              disable_objective_call=False,
              absolute_gap=1e-8,
              relative_gap=1e-4,
              scale_function=_default_scale,
              queue_tolerance=0,
              branch_tolerance=0,
              comparison_tolerance=0,
              objective_stop=None,
              bound_stop=None,
              node_limit=None,
              time_limit=None,
              initialize_queue=None,
              queue_strategy="bound",
              log_interval_seconds=1.0,
              log_new_incumbent=True,
              log=_notset):
        """Solve a problem using branch-and-bound.

        Note
        ----
        Parameters for this function are treated differently
        depending on whether the process is a worker or
        dispatcher. For the serial case (no MPI), the single
        process is both a worker and a dispatcher. For the
        parallel case, exactly one process is a dispatcher
        and all processes are workers. A **(W)** in the
        parameter description indicates that it is only used
        by worker processes (ignored otherwise). A **(D)**
        in the parameter description indicates that it is
        only used by the dispatcher process (ignored
        otherwise). An **(A)** indicates that it is used by
        all processes, and it is assumed the same value is
        provided for each process; otherwise, the behavior
        is undefined.

        Parameters
        ----------
        problem : :class:`pybnb.Problem <pybnb.problem.Problem>`
            An object defining a branch-and-bound problem.
        best_objective : float, optional
            Initializes the solve with an assumed best
            objective. This is the only option used by both
            worker and dispatcher processes that can be set
            to a different value on each process. The
            dispatcher will collect all values and use the
            best. (default: None)
        disable_objective_call : bool, optional
            **(W)** Disables requests for an objective value from
            subproblems. (default: False)
        absolute_gap : float, optional
            **(A)** The maximum absolute difference between
            the global bound and best objective for the
            problem to be considered solved to
            optimality. Setting to `None` will disable this
            optimality check. (default: 1e-8)
        relative_gap : float, optional
            **(A)** The maximum relative difference (absolute
            difference scaled by `max{1.0,|objective|}`)
            between the global bound and best objective for
            the problem to be considered solved to
            optimality. Setting to `None` will disable this
            optimality check. (default: 1e-4)
        scale_function : function, optional
            **(A)** A function with signature `f(bound,
            objective) -> float` that returns a positive
            scale factor used to convert the absolute
            difference between the bound and objective into
            a relative difference. The relative difference
            is compared with the `relative_gap` convergence
            tolerance to determine if the solver should
            terminate. The default is equivalent to
            `max{1.0,|objective|}`.  Other examples one
            could use are `max{|bound|,|objective|}`,
            `(|bound|+|objective|)/2`, etc.
        queue_tolerance : float, optional
            **(A)** The absolute tolerance used when
            deciding if a node is eligible to enter the
            queue. The difference between the node bound and
            the incumbent objective must be greater than
            this value. The default setting of zero means
            that nodes whose bound is equal to the incumbent
            objective are not eligible to enter the
            queue. Setting this to larger values can be used
            to control the queue size, but it should be kept
            small enough to allow absolute and relative
            optimality tolerances to be met. This option can
            also be set to `None` to allow nodes with a
            bound equal to (but not greater than) the
            incumbent objective to enter the queue.
            (default: 0)
        branch_tolerance : float, optional
            **(A)** The absolute tolerance used when
            deciding if the computed objective and bound for
            a node are sufficiently different to branch into
            the node. The default value of zero means that
            branching will occur if the bound is not exactly
            equal to the objective. This option can be set
            to `None` to enable branching for nodes with a
            bound and objective that are exactly
            equal. (default: 0)
        comparison_tolerance : float, optional
            **(A)** The absolute tolerance used when
            deciding if two objective or bound values are
            sufficiently different to be considered improved
            or worsened. This tolerance controls when the
            solver considers a new incumbent objective to be
            found. It also controls when warnings are output
            about bounds becoming worse on child
            nodes. Setting this to larger values can be used
            to avoid the above solver actions due to
            insignificant numerical differences, but it is
            better to deal with these numerical issues by
            rounding numbers to a reliable precision before
            returning them from the problem methods.
            (default: 0)
        objective_stop : float, optional
            **(A)** If provided, the solve will terminate
            when a feasible objective is found that is at
            least as good as the specified value, and the
            termination_condition flag on the results object
            will be set to "objective_limit". If this value
            is infinite, the solve will terminate as soon as
            a finite objective is found. (default: None)
        bound_stop : float, optional
            **(A)** If provided, the solve will terminate
            when the global bound on the objective is at
            least as good as the specified value, and the
            termination_condition flag on the results object
            will be set to "objective_limit". If this value
            is infinite, the solve will terminate as soon as
            a finite bound is found. (default: None)
        node_limit : int, optional
            **(D)** If provided, the solve will begin to
            terminate once this many nodes have been served
            from the dispatcher queue, and the
            termination_condition flag on the results object
            will be set to "node_limit". (default: None)
        time_limit : float, optional
            **(D)** If provided, the solve will begin to
            terminate once this amount of time has passed,
            and the termination_condition flag on the
            results object will be set to "time_limit". Note
            that the solve may run for an arbitrarily longer
            amount of time, depending how long worker
            processes spend completing their final
            task. (default: None)
        initialize_queue : :class:`pybnb.dispatcher.DispatcherQueueData`, optional
            **(D)** Initializes the dispatcher queue with
            that remaining from a previous solve (obtained
            by calling :func:`Solver.save_dispatcher_queue`
            after the solve). If left as None, the queue
            will be initialized with a single root node
            created by calling :func:`problem.save_state
            <pybnb.problem.Problem.save_state`.
            (default: None)
        queue_strategy : :class:`QueueStrategy <pybnb.common.QueueStrategy>`
            **(D)** Sets the strategy for prioritizing nodes
            in the central dispatcher queue. See the
            :class:`QueueStrategy
            <pybnb.common.QueueStrategy>` enum for the list
            of acceptable values. This keyword can be
            assigned one of the enumeration attributes or an
            equivalent string name. (default: "bound")
        log_interval_seconds : float, optional
            **(D)** The approximate time (in seconds)
            between solver log updates. More time may pass
            between log updates if no updates have been
            received from worker processes, and less time
            may pass if a new incumbent objective is
            found. (default: 1.0)
        log_new_incumbent : bool
            **(D)** Controls whether updates to the best
            objective are logged immediately (overriding the
            log interval). Setting this to false can be
            useful when frequent updates to the incumbent
            are expected and the additional logging slows
            down the dispatcher. (default: True)
        log : logging.Logger, optional
            **(D)** A log object where solver output should
            be sent. The default value causes all output to
            be streamed to the console. Setting to None
            disables all output.

        Returns
        -------
        results : :class:`SolverResults`
            An object storing information about the solve.
        """
        self._reset_local_solve_stats()
        self._solve_start = self._time()

        assert (initialize_queue is None) or \
            (self.is_dispatcher)

        if best_objective is None:
            best_objective = problem.infeasible_objective()

        results = SolverResults()
        convergence_checker = ConvergenceChecker(
            problem.sense(),
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            scale_function=scale_function,
            queue_tolerance=queue_tolerance,
            branch_tolerance=branch_tolerance,
            comparison_tolerance=comparison_tolerance,
            objective_stop=objective_stop,
            bound_stop=bound_stop)
        problem.notify_solve_begins(self.comm,
                                    self.worker_comm,
                                    convergence_checker)
        root = Node()
        root.queue_priority = 0
        problem.save_state(root)
        try:
            if self.is_dispatcher:
                if initialize_queue is None:
                    root.bound = problem.unbounded_objective()
                    root.objective = best_objective
                    assert root.tree_id is None
                    Node._insert_tree_id(root._data, 0)
                    initialize_queue = DispatcherQueueData(
                        nodes=[Node(data_=root._data.copy())],
                        next_tree_id=1)
                if log is _notset:
                    log = get_simple_logger()
                if type(queue_strategy) is \
                   QueueStrategy:
                    queue_strategy = \
                        queue_strategy.value
                self._disp.initialize(
                    best_objective,
                    initialize_queue,
                    queue_strategy,
                    convergence_checker,
                    node_limit,
                    time_limit,
                    log,
                    log_interval_seconds,
                    log_new_incumbent)
            if not self.is_worker:
                def handler(signum, frame):       #pragma:nocover
                    self._disp.log_warning(
                        "Solve interrupted by user. "
                        "Waiting for current worker "
                        "jobs to complete before "
                        "terminating the solve.")
                    self._disp.termination_condition = \
                        TerminationCondition.interrupted
                with MPI_InterruptHandler(handler):
                    tmp = self._disp.serve()
            else:
                def handler(signum, frame):       #pragma:nocover
                    if self.is_dispatcher:
                        self._disp.log_warning(
                            "Solve interrupted by user. "
                            "Waiting for current worker "
                            "jobs to complete before "
                            "terminating the solve.")
                        self._disp.termination_condition = \
                            TerminationCondition.interrupted
                with MPI_InterruptHandler(handler):
                    tmp = self._solve(problem,
                                      best_objective,
                                      disable_objective_call,
                                      convergence_checker,
                                      results)
                if not self.is_dispatcher:
                    self._disp.clear_cache()
            (results.objective,
             results.bound,
             results.termination_condition,
             self._global_solve_info) = tmp
            results.nodes = self._global_solve_info.explored_nodes_count
            self._fill_results(results, convergence_checker)
        except:                                        #pragma:nocover
            sys.stderr.write("Exception caught: "+str(sys.exc_info()[1])+"\n")
            sys.stderr.write("Attempting to shut down, but this may hang.\n")
            sys.stderr.flush()
            raise
        finally:
            problem.load_state(root)
        self._wall_time = self._time() - self._solve_start
        results.wall_time = self._wall_time

        assert results.solution_status in SolutionStatus,\
            str(results)
        assert results.termination_condition in TerminationCondition,\
            str(results)

        problem.notify_solve_finished(self.comm,
                                      self.worker_comm,
                                      results)
        if self.is_dispatcher and \
           (log is not None) and \
           (not log.disabled):
            self._disp.log_info("")
            if results.solution_status in ("feasible", "optimal"):
                agap = convergence_checker.compute_absolute_gap(
                    results.bound,
                    results.objective)
                rgap = convergence_checker.compute_relative_gap(
                    results.bound,
                    results.objective)
                if results.solution_status == "feasible":
                    self._disp.log_info("Feasible solution found")
                else:
                    if (convergence_checker.absolute_gap is not None) and \
                       agap <= convergence_checker.absolute_gap:
                        self._disp.log_info("Absolute optimality tolerance met")
                    if (convergence_checker.relative_gap is not None) and \
                       rgap <= convergence_checker.relative_gap:
                        self._disp.log_info("Relative optimality tolerance met")
                    assert results.solution_status == "optimal"
                    self._disp.log_info("Optimal solution found!")
            elif results.solution_status == "infeasible":
                self._disp.log_info("Problem is infeasible")
            elif results.solution_status == "unbounded":
                self._disp.log_info("Problem is unbounded")
            elif results.solution_status == "invalid":      #pragma:nocover
                self._disp.log_info("Problem is invalid")
            else:
                assert results.solution_status == "unknown"
                self._disp.log_info("Status unknown")
            self._disp.log_info("")
            self._disp.log_info(str(results))

        return results

def summarize_worker_statistics(stats, stream=sys.stdout):
    """Writes a summary of workers statistics to an
    output stream.

    Parameters
    ----------
    stats : dict
        A dictionary of worker statistics returned from
        a call to :func:`collect_worker_statistics`.
    stream : file-like object, or string, optional
        A file-like object or a filename where results
        should be written to. (default: ``sys.stdout``)
    """
    import numpy
    wall_time = numpy.array(stats['wall_time'],
                            dtype=float)
    queue_time = numpy.array(stats['queue_time'],
                             dtype=float)
    queue_count = numpy.array(stats['queue_call_count'],
                              dtype=int)
    objective_time = numpy.array(stats['objective_time'],
                                 dtype=float)
    objective_count = numpy.array(stats['objective_call_count'],
                                  dtype=int)
    bound_time = numpy.array(stats['bound_time'],
                             dtype=float)
    bound_count = numpy.array(stats['bound_call_count'],
                              dtype=int)
    branch_time = numpy.array(stats['branch_time'],
                              dtype=float)
    branch_count = numpy.array(stats['branch_call_count'],
                               dtype=int)
    load_state_time = numpy.array(stats['load_state_time'],
                                  dtype=float)
    load_state_count = numpy.array(stats['load_state_call_count'],
                                   dtype=int)
    explored_nodes_count = numpy.array(stats['explored_nodes_count'],
                                       dtype=int)
    work_time = wall_time - queue_time

    with as_stream(stream) as stream:
        stream.write("Number of Workers:   %6d\n"
                     % (len(wall_time)))
        div = max(1.0,numpy.mean(explored_nodes_count))
        numerator = numpy.max(explored_nodes_count) - \
            numpy.min(explored_nodes_count)
        if explored_nodes_count.sum() == 0:
            stream.write("Load Imbalance:     %6.2f%%\n"
                         % (0.0))
        else:
            stream.write("Load Imbalance:     %6.2f%%\n"
                         % (numerator/div*100.0))
            stream.write(" - min: %d\n" % (numpy.min(explored_nodes_count)))
            stream.write(" - max: %d\n" % (numpy.max(explored_nodes_count)))
        stream.write("Average Worker Timing:\n")
        queue_count_str = "%d" % queue_count.sum()
        tmp = "%"+str(len(queue_count_str))+"d"
        bound_count_str = tmp % bound_count.sum()
        objective_count_str = tmp % objective_count.sum()
        branch_count_str = tmp % branch_count.sum()
        load_state_count_str = tmp % load_state_count.sum()
        div1 = numpy.copy(wall_time)
        div1[div1 == 0] = 1
        div2 = numpy.copy(queue_count)
        div2[div2 == 0] = 1
        stream.write(" - queue:     %6.2f%% [avg time: %8s, count: %s]\n"
                     % (numpy.mean(queue_time/div1)*100.0,
                        time_format(numpy.mean(queue_time/div2),
                                    align_unit=True),
                        queue_count_str))
        div2 = numpy.copy(load_state_count)
        div2[div2==0] = 1
        stream.write(" - load_state:%6.2f%% [avg time: %8s, count: %s]\n"
                     % (numpy.mean((load_state_time/div1))*100.0,
                        time_format(numpy.mean(load_state_time/div2),
                                    align_unit=True),
                        load_state_count_str))
        div2 = numpy.copy(bound_count)
        div2[div2==0] = 1
        stream.write(" - bound:     %6.2f%% [avg time: %8s, count: %s]\n"
                     % (numpy.mean((bound_time/div1))*100.0,
                        time_format(numpy.mean(bound_time/div2),
                                    align_unit=True),
                        bound_count_str))
        div2 = numpy.copy(objective_count)
        div2[div2==0] = 1
        stream.write(" - objective: %6.2f%% [avg time: %8s, count: %s]\n"
                     % (numpy.mean((objective_time/div1))*100.0,
                        time_format(numpy.mean(objective_time/div2),
                                    align_unit=True),
                        objective_count_str))
        div2 = numpy.copy(branch_count)
        div2[div2==0] = 1
        stream.write(" - branch:    %6.2f%% [avg time: %8s, count: %s]\n"
                     % (numpy.mean((branch_time/div1))*100.0,
                        time_format(numpy.mean(branch_time/div2),
                                    align_unit=True),
                        branch_count_str))
        other_time = work_time - objective_time - bound_time - branch_time - load_state_time
        div2 = numpy.copy(queue_count)
        div2[div2 == 0] = 1
        stream.write(" - other:     %6.2f%% [avg time: %8s, count: %s]\n"
                     % (numpy.mean(other_time/div1)*100.0,
                        time_format(numpy.mean(other_time/div2),
                                    align_unit=True),
                        queue_count_str))


def solve(problem,
          comm=_notset,
          dispatcher_rank=0,
          log_filename=None,
          results_filename=None,
          **kwds):
    """Solves a branch-and-bound problem and returns the
    solution.

    Note
    ----
    This function also collects and summarizes runtime
    workload statistics, which may introduce additional
    overhead. This overhead can be avoided by directly
    instantiating a :class:`Solver` object and
    calling the :func:`Solver.solve` method.

    Parameters
    ----------
    problem : :class:`pybnb.Problem <pybnb.problem.Problem>`
        An object that defines a branch-and-bound problem
    comm : ``mpi4py.MPI.Comm``, optional
        The MPI communicator to use. If unset, the
        mpi4py.MPI.COMM_WORLD communicator will be
        used. Setting this keyword to None will disable the
        use of MPI and avoid an attempted import of
        mpi4py.MPI (which avoids triggering a call to
        `MPI_Init()`).
    dispatcher_rank : int, optional
        The process with this rank will be designated the
        dispatcher process. If MPI functionality is disabled
        (by setting comm=None, or when comm.size==1), this
        keyword must be left at 0. (default: 0)
    log_filename : string, optional
        A filename where solver output should be sent in
        addition to console. This keyword will be ignored if
        the `log` keyword is set. (default: None)
    results_filename : string, optional
        Saves the solver results into a YAML-formatted file
        with the given name. (default: None)
    **kwds
        Additional keywords to be passed to
        :func:`Solver.solve`. See that method for additional
        keyword documentation.

    Returns
    -------
    results : :class:`SolverResults`
        An object storing information about the solve.
    """

    opt = Solver(comm=comm,
                 dispatcher_rank=dispatcher_rank)

    if (opt.is_dispatcher) and \
       ("log" not in kwds) and \
       (log_filename is not None):
        kwds["log"] = get_simple_logger(
            filename=log_filename)

    results = opt.solve(problem, **kwds)

    stats = opt.collect_worker_statistics()
    if opt.is_dispatcher:
        tmp = six.StringIO()
        summarize_worker_statistics(stats, stream=tmp)
        opt._disp.log_info(tmp.getvalue())

    if opt.is_dispatcher and (results_filename is not None):
        results.write(results_filename)

    return results
