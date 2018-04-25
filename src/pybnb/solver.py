import sys
import time

from pybnb.misc import (metric_fmt,
                          as_stream,
                          get_simple_logger)
from pybnb.problem import (GenericProblem,
                             ProblemNode)
from pybnb.dispatcher_proxy import DispatcherProxy
from pybnb.dispatcher import (Dispatcher,
                                TreeIdLabeler,
                                SavedDispatcherQueue)

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

import six
from six import StringIO

class _notset(object):
    pass

class SolverResults(object):
    """Stores the results of a branch-and-bound solve."""

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

        Args:
            stream: A file-like object or a filename where
                results should be written
                to. (default=sys.stdout)
        """
        with as_stream(stream) as stream:
            stream.write("solver results:\n")
            self.write(stream, prefix=" - ", pretty=True)

    def write(self, stream, prefix="", pretty=False):
        """Writes results in YAML format to a stream or
        file.

        Args:
            stream: A file-like object or a filename where
                results should be written to.
            prefix (`str`): A string to prefix with each line
                that is written. (default='')
            pretty: Indicates whether or not certain values
                should be formatted for more human-readable
                output. (default=False)
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
                            val = metric_fmt(val, unit='s')
                        elif name in ('objective','bound',
                                      'absolute_gap','relative_gap'):
                            val = "%.7g" % (val)
                stream.write(prefix+'%s: %s\n'
                             % (name, val))
            for name in names:
                stream.write(prefix+'%s: %s\n'
                              % (name, getattr(self, name)))

    def __str__(self):
        """Represents the results as a string."""
        tmp = StringIO()
        self.pprint(stream=tmp)
        return tmp.getvalue()

class Solver(object):
    """A parallel branch-and-bound solver."""

    @staticmethod
    def summarize_worker_statistics(stats, stream=sys.stdout):
        """Writes a summary of workers statistics to an
        output stream.

        Args:
            stats (`dict`): A dictionary of worker statistics
                returned from a call to collect_worker_statics()
                on a solver.
            stream: A file-like object or a filename where
                results should be written to. (default=sys.stdout)
        """
        import numpy
        explored_nodes_count = numpy.array(stats['explored_nodes_count'],
                                           dtype=int)
        wall_time = numpy.array(stats['wall_time'],
                                dtype=float)
        objective_eval_time = numpy.array(stats['objective_eval_time'],
                                          dtype=float)
        objective_eval_count = numpy.array(stats['objective_eval_count'],
                                           dtype=int)
        bound_eval_time = numpy.array(stats['bound_eval_time'],
                                      dtype=float)
        bound_eval_count = numpy.array(stats['bound_eval_count'],
                                       dtype=int)
        comm_time = numpy.array(stats['comm_time'], dtype=float)
        work_time = wall_time - comm_time

        with as_stream(stream) as stream:
            stream.write("Number of Workers:   %6d\n"
                         % (len(wall_time)))
            div = float(max(1,float(explored_nodes_count.sum())))
            stream.write("Average Work Load:   %6.2f%%\n"
                         % (numpy.mean(explored_nodes_count/div)*100.0))
            div = max(1.0,numpy.mean(explored_nodes_count))
            if explored_nodes_count.sum() == 0:
                stream.write("Work Load Imbalance: %6.2f%%\n"
                             % (0.0))
            else:
                stream.write("Work Load Imbalance: %6.2f%%\n"
                             % ((numpy.max(explored_nodes_count)/div - 1.0)*100.0))
            stream.write("Average Worker Timing:\n")
            div = numpy.copy(wall_time)
            div[div == 0] = 1
            stream.write(" - communication: %6.2f%%\n"
                         % (numpy.mean(comm_time/div)*100.0))
            stream.write(" - work:          %6.2f%%\n"
                         % (numpy.mean(work_time/div)*100.0))
            div1 = numpy.copy(work_time)
            div1[div1==0] = 1
            div2 = numpy.copy(objective_eval_count)
            div2[div2==0] = 1
            stream.write("   - objective eval: %6.2f%% (avg time=%s, count=%d)\n"
                         % (numpy.mean((objective_eval_time/div1))*100.0,
                            metric_fmt(numpy.mean(objective_eval_time/div2), unit='s'),
                            objective_eval_count.sum()))
            div2 = numpy.copy(bound_eval_count)
            div2[div2==0] = 1
            stream.write("   - bound eval:     %6.2f%% (avg time=%s, count=%d)\n"
                         % (numpy.mean((bound_eval_time/div1))*100.0,
                            metric_fmt(numpy.mean(bound_eval_time/div2), unit='s'),
                            bound_eval_count.sum()))
            stream.write("   - other:          %6.2f%%\n"
                         % (numpy.mean((work_time - objective_eval_time - bound_eval_time) / \
                                       div1)*100.0))

    def __init__(self,
                 comm=_notset,
                 dispatcher_rank=0):
        """"
        Args:
            comm: The MPI communicator to use. If left to
                its default value, the mpi4py.MPI.COMM_WORLD
                communicator will be used. This keyword can
                also be set to None to disable MPI
                completely (and avoid an attempted import of
                mpi4py.MPI).
            dispatcher_rank (int): The process with this
                rank will be designated as the dispatcher
                process. If MPI functionality is disabled
                (i.e., comm=None), this keyword must be
                0. (default=0)
        """
        mpi = True
        if comm is None:
            mpi = False
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
            if comm.size == 1:
                mpi = False
                comm = None
        if (not mpi) and \
           ((comm is not None) or (dispatcher_rank != 0)):
            raise ValueError("MPI functionality has been disabled but "
                             "the 'comm' keyword is set to something "
                             "other than None or the 'dispatcher_rank' "
                             "keyword is set to something other than 0.")
        if mpi:
            if (int(dispatcher_rank) != dispatcher_rank) or \
               (dispatcher_rank < 0) or \
               (dispatcher_rank >= comm.size):
                raise ValueError("The 'dispatcher_rank' keyword "
                                 "has been set to %s, which is not "
                                 "an available rank given the "
                                 "size of the MPI communicator (%d)."
                                 % (dispatcher_rank, comm.size))
            dispatcher_rank = int(dispatcher_rank)
            if comm.rank == dispatcher_rank:
                self._disp = Dispatcher(comm.Dup())
                self._worker_flag = False
                self._dispatcher_flag = True
            else:
                self._disp = DispatcherProxy(comm.Dup())
                self._worker_flag = True
                self._dispatcher_flag = False
            self._time = mpi4py.MPI.Wtime
        else:
            assert comm is None
            assert dispatcher_rank == 0
            self._disp = Dispatcher(None)
            self._worker_flag = True
            self._dispatcher_flag = True
            self._time = time.time
        assert self._worker_flag in (True, False)
        assert self._dispatcher_flag in (True, False)
        assert self._disp is not None
        assert self._time is not None
        self._wall_time = None
        self._objective_eval_time = None
        self._objective_eval_count = None
        self._bound_eval_time = None
        self._bound_eval_count = None
        self._explored_nodes_count = None
        self._best_objective = None

    @property
    def worker(self):
        """Indicates if this process has been designated as
        a worker."""
        return self._worker_flag

    @property
    def dispatcher(self):
        """Indicates if this process has been designated as
        the dispatcher."""
        return self._dispatcher_flag

    @property
    def comm(self):
        """The full MPI communicator that includes the
        dispatcher and all workers. Will be None if MPI
        functionality has been disabled."""
        return self._disp.comm

    @property
    def worker_comm(self):
        """The worker MPI communicator. Will be None if this
        process is designated as a dispatcher or if MPI
        functionality has been disabled."""
        if not self.dispatcher:
            return self._disp.worker_comm
        return None

    @property
    def root_worker(self):
        """Indicates if this process has been designated as
        the root worker by the dispatcher."""
        if self.comm is not None:
            return (self._disp.root_worker_comm_rank == \
                    self.comm.rank)
        return True

    def _reset_local_solve_stats(self):
        self._wall_time = 0.0
        self._objective_eval_time = 0.0
        self._objective_eval_count = 0
        self._bound_eval_time = 0.0
        self._bound_eval_count = 0
        self._explored_nodes_count = 0
        self._best_objective = None
        if not self.dispatcher:
            self._disp.comm_time = 0.0

    def collect_worker_statistics(self):
        """Collect individual worker statistics about the
        most recent solve.

        Returns:
            a dictionary whose keys are the different \
            statistics collected
        """
        stats = {}
        if self.comm is not None:
            if self.worker:
                assert self.worker_comm is not None
                assert not self.dispatcher
                stats['wall_time'] = self.worker_comm.allgather(
                    self._wall_time)
                stats['objective_eval_time'] = self.worker_comm.allgather(
                    self._objective_eval_time)
                stats['objective_eval_count'] = self.worker_comm.allgather(
                    self._objective_eval_count)
                stats['bound_eval_time'] = self.worker_comm.allgather(
                    self._bound_eval_time)
                stats['bound_eval_count'] = self.worker_comm.allgather(
                    self._bound_eval_count)
                stats['explored_nodes_count'] = self.worker_comm.allgather(
                    self._explored_nodes_count)
                stats['comm_time'] = self.worker_comm.allgather(
                    self._disp.comm_time)
                if self.root_worker:
                    self.comm.send(stats, self._disp.dispatcher_rank)
            else:
                assert self.worker_comm is None
                assert self.dispatcher
                stats = self.comm.recv(source=self._disp.root_worker_comm_rank)
        else:
            assert self.worker_comm is None
            stats['wall_time'] = [self._wall_time]
            stats['objective_eval_time'] = [self._objective_eval_time]
            stats['objective_eval_count'] = [self._objective_eval_count]
            stats['bound_eval_time'] = [self._bound_eval_time]
            stats['bound_eval_count'] = [self._bound_eval_count]
            stats['explored_nodes_count'] = [self._explored_nodes_count]
            stats['comm_time'] = [0.0]

        return stats

    def save_dispatcher_queue(self):
        """If this process is the dispatcher and there are
        any nodes remaining in the queue after the most
        recent solve, this method returns an object that can
        be used to reinitialize the queue in a new solve by
        assigning it to the `initialize_queue` keyword for
        the :attr:`solve` method. If this process is not the
        dispatcher or the queue is empty, this method
        returns None."""
        dispatcher_queue = None
        if self.dispatcher and \
           (self._disp.queue.qsize() > 0):
            dispatcher_queue = self._disp.save_dispatcher_queue()
        return dispatcher_queue

    def solve(self,
              problem,
              best_objective=None,
              initialize_queue=None,
              absolute_gap=1e-8,
              relative_gap=1e-4,
              cutoff=None,
              node_limit=None,
              time_limit=None,
              absolute_tolerance=1e-10,
              log_interval_seconds=1.0,
              log=_notset):
        """Solves a user-defined instance of type
        :class:`pybnb.problem.Problem` using the
        branch-and-bound algorithm.

        Args:
            problem: A user-defined instance of type
                :class:`pybnb.problem.Problem`.
            best_objective (`float`): Initializes the solve
                with an assumed best
                objective. (default=None)
            initialize_queue: Can be assigned the return
                value of a call to the
                :attr:`save_dispatcher_queue` method after a
                previous solve to initialize the current
                solve with any nodes remaining in the queue
                after the previous
                solve. (default=None)
            absolute_gap (`float`): The solver will terminate
                with an optimal status when the absolute gap
                between the objective and bound is less than
                this value. (default=1e-8)
            relative_gap (`float`): The solver will terminate
                with an optimal status when the relative gap
                between the objective and bound is less than
                this value. (default=1e-4)
            cutoff (`float`): If provided, when the best
                objective is proven worse than this value,
                the solver will begin to terminate, and the
                termination_condition flag will be set to
                'cutoff'. (default=None)
            node_limit (`int`): If provided, the solver will
                begin to terminate once this many nodes have
                been processed. It is possible that more
                nodes will be processed when running in
                parallel mode, but not by more than the
                number of available workers. If this setting
                initiates a shutdown, then the
                termination_condition flag will be set to
                'node_limit'. (default=None)
            time_limit (`float`): If provided, the solver will
                begin to terminate the solve once this
                amount of time has passed. The solver may
                run for an arbitrarily longer amount of
                time, depending how long workers spend
                processing their current node. If this
                setting initiates a shutdown, then the
                termination_condition flag will be set to
                'time_limit'. (default=None)
            absolute_tolerance (`float`): The absolute
                tolerance use when deciding if two objective
                values are sufficiently
                different. (default=1e-10)
            log_interval_seconds (`float`): The approximate
                maximum time (in seconds) between solver log
                updates. More time may pass between log
                updates if no updates have been received
                from any workers, and less time may pass if
                a new incumbent is found. (default=1.0)
            log: A logging.Logger object where solver output
                should be sent. The default value causes all
                output to be streamed to the
                console. Setting to None disables all
                output.

        Returns:
            A pybnb.solver.SolverResults object.
        """

        if best_objective is None:
            best_objective = problem.infeasible_objective

        # broadcast options from dispatcher to everyone else
        # to ensure consistency
        if self.comm is not None:
            (best_objective, absolute_gap, relative_gap,
             cutoff, absolute_tolerance) = \
                self.comm.bcast((best_objective,
                                 absolute_gap, relative_gap,
                                 cutoff, absolute_tolerance),
                                root=self._disp.dispatcher_rank)
            if not self.dispatcher:
                # These are not used unless this process is
                # the dispatcher
                node_limit = None
                time_limit = None
                log_interval_seconds = None
                log = None
                if initialize_queue is not None:       #pragma:nocover
                    raise ValueError("The 'initialize_queue' keyword "
                                     "must be None for all processes "
                                     "except the dispatcher.")

        results = SolverResults()
        generic_problem = GenericProblem(problem.sense,
                                         absolute_gap=absolute_gap,
                                         relative_gap=relative_gap,
                                         absolute_tolerance=absolute_tolerance,
                                         cutoff=cutoff)
        root = problem.new_node()
        problem.save_state(root)
        self._reset_local_solve_stats()
        start = self._time()
        try:
            if self.dispatcher:
                if initialize_queue is None:
                    root.bound = problem.unbounded_objective
                    tree_id_labeler = TreeIdLabeler()
                    if root.tree_id is None:
                        root.tree_id = tree_id_labeler()
                    initialize_queue = SavedDispatcherQueue(
                        states=[root._state],
                        tree_id_labeler=tree_id_labeler)
                if log is _notset:
                    log = get_simple_logger(show=True)
                elif log is None:
                    log = get_simple_logger(show=False)
                self._disp.initialize(
                    best_objective,
                    initialize_queue,
                    generic_problem,
                    node_limit,
                    time_limit,
                    log,
                    log_interval_seconds)
            if self.comm is not None:
                self.comm.Barrier()
            if not self.worker:
                self._disp.serve()
            else:
                self._solve(problem,
                            best_objective,
                            generic_problem,
                            results)
        except:                                        #pragma:nocover
            sys.stderr.write("Exception caught: "+str(sys.exc_info()[1])+"\n")
            sys.stderr.write("Attempting to shut down, but this may hang.\n")
            sys.stderr.flush()
            raise
        finally:
            problem.load_state(root)
        if self.worker:
            self._disp.barrier()
            if self.root_worker:
                self._disp.solve_finished()
        stop = self._time()
        self._wall_time = stop-start
        if self.comm is not None:
            if self.worker:
                results.nodes = self.worker_comm.allreduce(
                    self._explored_nodes_count,
                    op=mpi4py.MPI.SUM)
            results.wall_time = self.comm.allreduce(
                self._wall_time,
                op=mpi4py.MPI.MAX)
            if self.root_worker:
                assert not self.dispatcher
                self.comm.send(results, self._disp.dispatcher_rank)
            elif self.dispatcher:
                results = self.comm.recv(source=self._disp.root_worker_comm_rank)
                results.termination_condition = self._disp.get_termination_condition()
            results.termination_condition = self.comm.bcast(results.termination_condition,
                                                            root=self._disp.dispatcher_rank)
        else:
            results.nodes = self._explored_nodes_count
            results.wall_time = self._wall_time
            results.termination_condition = self._disp.get_termination_condition()

        assert results.solution_status in ("optimal",
                                           "feasible",
                                           "infeasible",
                                           "unbounded",
                                           "unknown"), str(results)
        assert results.termination_condition in ("optimality",
                                                 "feasibilty",
                                                 "cutoff",
                                                 "node_limit",
                                                 "time_limit",
                                                 "no_nodes"), str(results)
        problem.notify_solve_finished(self.comm,
                                      self.worker_comm,
                                      results)
        if self.dispatcher:
            self._disp.log_info("")
            if results.solution_status in ("feasible", "optimal"):
                agap = generic_problem.compute_absolute_gap(
                    results.bound,
                    results.objective)
                rgap = generic_problem.compute_relative_gap(
                    results.bound,
                    results.objective)
                if results.solution_status == "feasible":
                    self._disp.log_info("Feasible solution found")
                else:
                    if agap < generic_problem.absolute_gap_tolerance:
                        self._disp.log_info("Absolute optimality tolerance met")
                    if rgap < generic_problem.relative_gap_tolerance:
                        self._disp.log_info("Relative optimality tolerance met")
                    assert results.solution_status == "optimal"
                    self._disp.log_info("Optimal solution found")
                self._disp.log_info(" - absolute gap: %.6g"
                                    % (agap))
                self._disp.log_info(" - relative gap: %.6g"
                                    % (rgap))
            elif results.solution_status == "infeasible":
                self._disp.log_info("Problem is infeasible")
            elif results.solution_status == "unbounded":
                self._disp.log_info("Problem is unbounded")
            else:
                assert results.solution_status == "unknown"
                self._disp.log_info("Status unknown")
            self._disp.log_info("")
            self._disp.log_info(str(results))

        return results

    def _check_update_best_objective(self,
                                     generic_problem,
                                     new_objective):
        if generic_problem.objective_improved(new_objective,
                                              self._best_objective):
            self._best_objective = new_objective
            return True
        else:
            return False

    def _solve(self,
               problem,
               best_objective,
               generic_problem,
               results):
        assert problem.sense == generic_problem.sense
        infeasible_objective = problem.infeasible_objective
        assert infeasible_objective == generic_problem.infeasible_objective
        unbounded_objective = problem.unbounded_objective
        assert unbounded_objective == generic_problem.unbounded_objective

        self._best_objective = best_objective
        child_states = ()
        bound = problem.unbounded_objective
        working_node = problem.new_node()
        assert working_node.tree_id is None
        # start the work loop
        while (1):
            new_objective, state = \
                self._disp.update(self._best_objective,
                                  bound,
                                  self._explored_nodes_count,
                                  child_states)

            updated = self._check_update_best_objective(
                generic_problem,
                new_objective)
            if updated:
                problem.notify_new_best_objective_received(
                    self.worker_comm,
                    self._best_objective)
            del updated

            child_states = ()

            if state is None:
                # make sure all processes have the exact same best
                # objective value (not just subject to tolerances)
                self._best_objective = new_objective
                break
            del new_objective

            working_node._state = state
            bound = working_node.bound
            assert working_node.tree_id is not None
            self._explored_nodes_count += 1

            # check the weak bound before loading the problem
            # state, to avoid doing that work if possible
            if (bound != infeasible_objective) and \
               generic_problem.objective_can_improve(
                   self._best_objective,
                   bound) and \
                (not generic_problem.cutoff_is_met(bound)):

                problem.load_state(working_node)
                del state

                bound_eval_start = self._time()
                new_bound = problem.bound()
                self._bound_eval_time += self._time()-bound_eval_start
                self._bound_eval_count += 1
                if generic_problem.bound_worsened(new_bound, bound):    #pragma:nocover
                    self._disp.log_warning("WARNING: Bound became worse "
                                           "(old=%r, new=%r)"
                                           % (bound, new_bound))
                working_node.bound = new_bound
                bound = new_bound

            if (bound != infeasible_objective) and \
                generic_problem.objective_can_improve(
                    self._best_objective,
                    bound) and \
                (not generic_problem.cutoff_is_met(bound)):
                objective_eval_start = self._time()
                obj = problem.objective()
                self._objective_eval_time += self._time()-objective_eval_start
                self._objective_eval_count += 1
                if obj is not None:
                    if generic_problem.bound_is_suboptimal(bound, obj): #pragma:nocover
                        self._disp.log_warning(
                            "WARNING: Local node bound is worse "
                            "than local node objective (bound=%r, "
                            "objective=%r)" % (bound, obj))
                    updated = self._check_update_best_objective(
                        generic_problem,
                        obj)
                    if updated:
                        problem.notify_new_best_objective(
                            self.worker_comm,
                            self._best_objective)
                    del updated
                if (obj != generic_problem.unbounded_objective) and \
                   generic_problem.objective_can_improve(
                       self._best_objective,
                       bound):
                    child_states = [child._state for child in
                                    problem.branch(working_node)]
        results.objective = self._best_objective
        results.bound = self._disp.finalize()

        if results.bound == infeasible_objective:
            assert results.objective == infeasible_objective, \
                str(results.objective)
            results.solution_status = "infeasible"
        elif results.objective == infeasible_objective:
            results.solution_status = "unknown"
        elif results.objective == unbounded_objective:
            assert results.bound == unbounded_objective, \
                str(results.bound)
            results.solution_status = "unbounded"
        else:
            results.absolute_gap = generic_problem.\
                                   compute_absolute_gap(results.bound,
                                                        results.objective)
            results.relative_gap = generic_problem.\
                                   compute_relative_gap(results.bound,
                                                        results.objective)
            if generic_problem.objective_is_optimal(results.objective,
                                                    results.bound):
                results.solution_status = "optimal"
            else:
                results.solution_status = "feasible"
