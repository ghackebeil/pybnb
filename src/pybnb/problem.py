"""
Branch-and-bound problem definition.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

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
        zero or more children.

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
        root node, the :attr:`tree_id <pybnb.node.Node.tree_id`
        and :attr:`parent_tree_id <pybnb.node.Node.parent_tree_id`
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

    def notify_new_best_objective_received(self,
                                           worker_comm,
                                           best_objective):
        """Called when a branch-and-bound solver receives a
        new best objective. The :class:`Problem
        <pybnb.problem.Problem>` base class provides a
        default implementation for this method that does
        nothing.

        Parameters
        ----------
        worker_comm : ``mpi4py.MPI.Comm``
            The MPI communicator to used by the
            workers. Will be None if MPI has been disabled.
        best_objective : float
            The new best objective value.
        """
        pass

    def notify_new_best_objective(self,
                                  worker_comm,
                                  best_objective):
        """Called when a branch-and-bound solver locally
        computes a new best objective. The :class:`Problem
        <pybnb.problem.Problem>` base class provides a
        default implementation for this method that does
        nothing.

        Parameters
        ----------
        worker_comm : ``mpi4py.MPI.Comm``
            The MPI communicator to used by the
            workers. Will be None if MPI has been disabled.
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
        comm : ``mpi4py.MPI.Comm``
            The full MPI communicator to used by all
            processes.  Will be None if MPI has been
            disabled.
        worker_comm : ``mpi4py.MPI.Comm``
            The MPI communicator to used by the
            workers. Will be None if MPI has been disabled.
        results : :class:`pybnb.solver.SolverResults`
            The fully populated results container that will
            be returned from the solver.
        """
        pass
