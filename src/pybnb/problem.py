"""
Branch-and-bound problem definition.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import array

from pybnb.common import (minimize,
                          maximize,
                          inf)

class Problem(object):
    """The abstract base class used for defining
    branch-and-bound problems."""
    __slots__ = ()

    def infeasible_objective(self):
        """Returns the value that represents an infeasible
        objective (i.e., +inf or -inf depending on the
        sense). The :class:`Problem <pybnb.problem.Problem>`
        base class implements this method. """
        if self.sense() == minimize:
            return inf
        else:
            assert self.sense() == maximize
            return -inf

    def unbounded_objective(self):
        """Returns the value that represents an unbounded
        objective (i.e., +inf or -inf depending on the
        sense). The :class:`Problem <pybnb.problem.Problem>`
        base class implements this method."""
        if self.sense() == minimize:
            return -inf
        else:
            assert self.sense() == maximize
            return inf

    #
    # Abstract Methods
    #

    def sense(self):                              #pragma:nocover
        """Returns the objective sense for this problem.

        Note
        ----
        This method is abstract and must be defined by the
        user.
        """
        raise NotImplementedError()

    def objective(self):                          #pragma:nocover
        """Returns a feasible value for the objective of the
        current problem state or
        :func:`self.infeasible_objective()
        <pybnb.problem.Problem.infeasible_objective>` if the
        current state is not feasible.

        Note
        ----
        This method is abstract and must be defined by the
        user.
        """
        raise NotImplementedError()

    def bound(self):                              #pragma:nocover
        """Returns a value that is a bound on the objective
        of the current problem state or
        :func:`self.unbounded_objective()
        <pybnb.problem.Problem.unbounded_objective>` if no
        non-trivial bound is available.

        Note
        ----
        This method is abstract and must be defined by the
        user.
        """
        raise NotImplementedError()

    def branch(self, parent_node):                #pragma:nocover
        """Returns a list of :class:`pybnb.node.Node`
        objects that partition the parent node state into
        zero or more children. This method can also be
        defined as a generator.

        Note
        ----
        This method is abstract and must be defined by the
        user.
        """
        raise NotImplementedError()

    def save_state(self, node):                   #pragma:nocover
        """Saves the current problem state into the given
        :class:`pybnb.node.Node` object.

        This method is guaranteed to be called once at the
        start of the solve by all processes involved to
        collect the root node problem state, but it may be
        called additional times. When it is called for the
        root node, the :attr:`node.tree_id <pybnb.node.Node.tree_id>`
        and :attr:`node.parent_tree_id <pybnb.node.Node.parent_tree_id>`
        will both be None.

        Note
        ----
        This method is abstract and must be defined by the
        user.
        """
        raise NotImplementedError()

    def load_state(self, node):                   #pragma:nocover
        """Loads the problem state that is stored on the
        given :class:`pybnb.node.Node` object.

        Note
        ----
        This method is abstract and must be defined by the
        user.
        """
        raise NotImplementedError()

    #
    # Optional Abstract Methods
    #

    def notify_solve_begins(self,
                            comm,
                            worker_comm,
                            convergence_checker):
        """Called when a branch-and-bound solver
        begins as solve. The :class:`Problem
        <pybnb.problem.Problem>` base class provides a
        default implementation for this method that does
        nothing.

        Parameters
        ----------
        comm : ``mpi4py.MPI.Comm`` or ``None``
            The full MPI communicator that includes all
            processes. Will be None if MPI has been
            disabled.
        worker_comm : ``mpi4py.MPI.Comm`` or ``None``
            The MPI communicator that includes only worker
            processes. Will be None if MPI has been
            disabled.
        convergence_checker : :class:`ConvergenceChecker <pybnb.convergence_checker.ConvergenceChecker>`:
            The class used for comparing the objective and
            bound values during the solve.
        """
        pass

    def notify_new_best_objective_received(self,
                                           best_objective):
        """Called when a branch-and-bound solver receives a
        new best objective. The :class:`Problem
        <pybnb.problem.Problem>` base class provides a
        default implementation for this method that does
        nothing.

        Parameters
        ----------
        best_objective : float
            The new best objective value.
        """
        pass

    def notify_new_best_objective(self,
                                  best_objective):
        """Called when a branch-and-bound solver locally
        computes a new best objective. The :class:`Problem
        <pybnb.problem.Problem>` base class provides a
        default implementation for this method that does
        nothing.

        Parameters
        ----------
        best_objective : float
            The new best objective value.
        """
        pass

    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        """Called when a branch-and-bound solver
        finishes. The :class:`Problem
        <pybnb.problem.Problem>` base class provides a
        default implementation for this method that does
        nothing.

        Parameters
        ----------
        comm : ``mpi4py.MPI.Comm`` or ``None``
            The full MPI communicator that includes all
            processes. Will be None if MPI has been
            disabled.
        worker_comm : ``mpi4py.MPI.Comm`` or ``None``
            The MPI communicator that includes only worker
            processes. Will be None if MPI has been
            disabled.
        results : :class:`pybnb.solver.SolverResults`
            The fully populated results container that will
            be returned from the solver.
        """
        pass

class _SolveInfo(object):
    __slots__ = ("data")
    _data_size = 9
    def __init__(self):
        self.data = array.array('d',[0]) * \
                    _SolveInfo._data_size

    def reset(self):
        """Resets all statistics to zero."""
        for i in range(len(self.data)):
            self.data[i] = 0.0

    def add_from(self, other):
        if type(other) is not _SolveInfo:
            raise TypeError("Type %s can not be added "
                            "with a _SolveInfo object"
                            % (other.__class__.__name__))
        assert len(self.data) == len(other.data)
        for i in range(_SolveInfo._data_size):
            self.data[i] += other.data[i]

    @property
    def explored_nodes_count(self):
        return int(self.data[0])
    @explored_nodes_count.setter
    def explored_nodes_count(self, val):
        self.data[0] = val

    def _increment_explored_nodes_count(self, val):
        self.data[0] += val

    @property
    def total_queue_time(self):
        return float(self.data[1])
    @total_queue_time.setter
    def total_queue_time(self, val):
        self.data[1] = val

    @property
    def queue_call_count(self):
        return int(self.data[2])
    @queue_call_count.setter
    def queue_call_count(self, val):
        self.data[2] = val

    def _increment_queue_stat(self, time_, count):
        self.data[1] += time_
        self.data[2] += count

    @property
    def total_objective_time(self):
        return float(self.data[3])
    @total_objective_time.setter
    def total_objective_time(self, val):
        self.data[3] = val

    @property
    def objective_call_count(self):
        return int(self.data[4])
    @objective_call_count.setter
    def objective_call_count(self, val):
        self.data[4] = val

    def _increment_objective_stat(self, time_, count):
        self.data[3] += time_
        self.data[4] += count

    @property
    def total_bound_time(self):
        return float(self.data[5])
    @total_bound_time.setter
    def total_bound_time(self, val):
        self.data[5] = val

    @property
    def bound_call_count(self):
        return int(self.data[6])
    @bound_call_count.setter
    def bound_call_count(self, val):
        self.data[6] = val

    def _increment_bound_stat(self, time_, count):
        self.data[5] += time_
        self.data[6] += count

    @property
    def total_branch_time(self):
        return float(self.data[7])
    @total_branch_time.setter
    def total_branch_time(self, val):
        self.data[7] = val

    @property
    def branch_call_count(self):
        return int(self.data[8])
    @branch_call_count.setter
    def branch_call_count(self, val):
        self.data[8] = val

    def _increment_branch_stat(self, time_, count):
        self.data[7] += time_
        self.data[8] += count

class _ProblemWithSolveInfoCollection(Problem):
    """A Problem objects that keeps track of statistics used
    by the solver"""

    def __init__(self):
        self._clock = None
        self._solve_info = None

    def set_clock(self, clock):
        self._clock = clock

    def set_solve_info_object(self, solve_info):
        assert isinstance(solve_info, _SolveInfo)
        self._solve_info = solve_info

class _SimpleSolveInfoCollector(_ProblemWithSolveInfoCollection):
    """A wrapper for Problem objects that collects statistics
    on methods called during the solve."""

    def __init__(self, problem):
        self._problem = problem
        super(_SimpleSolveInfoCollector, self).__init__()

    #
    # Define Problem methods
    #

    def sense(self):
        return self._problem.sense()

    def objective(self):
        start = self._clock()
        tmp = self._problem.objective()
        stop = self._clock()
        self._solve_info._increment_objective_stat(stop-start, 1)
        return tmp

    def bound(self):
        start = self._clock()
        tmp = self._problem.bound()
        stop = self._clock()
        self._solve_info._increment_bound_stat(stop-start, 1)
        return tmp

    def branch(self, parent_node):
        start = self._clock()
        for item in self._problem.branch(parent_node):
            yield item
        stop = self._clock()
        self._solve_info._increment_branch_stat(stop-start, 1)

    def save_state(self, node):
        self._problem.save_state(node)

    def load_state(self, node):
        self._problem.load_state(node)
        self._solve_info._increment_explored_nodes_count(1)

    def notify_solve_begins(self,
                            comm,
                            worker_comm,
                            convergence_checker):
        self._problem.notify_solve_begins(
            comm,
            worker_comm,
            convergence_checker)

    def notify_new_best_objective_received(self,
                                           best_objective):
        self._problem.notify_new_best_objective_received(
            best_objective)

    def notify_new_best_objective(self,
                                  best_objective):
        self._problem.notify_new_best_objective(
            best_objective)

    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        self._problem.notify_solve_finished(
            comm,
            worker_comm,
            results)
