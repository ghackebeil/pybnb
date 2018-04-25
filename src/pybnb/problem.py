import array

from pybnb.misc import (infinity,
                          is_infinite)

import numpy

minimize = (1,)
maximize = (-1,)

def new_storage_array(size):
    """Return a contiguous storage array of type double
    using the built-in array module"""
    #return array.array('d',[0])*size
    return numpy.empty(size, dtype=float)

class ProblemNode(object):
    __slots__ = ("_state",)

    _extra_state_slots = 6
    # state[-6] = best_objective
    # state[-5] = bound
    # state[-4] = tree_id
    # state[-3] = parent_tree_id
    # state[-2] = has_tree_id
    # state[-1] = tree_depth

    new_array = staticmethod(new_storage_array)

    def __init__(self, size=0, tree_depth=0):
        assert size >= 0
        assert tree_depth >= 0
        self._state = self.new_array(size + self._extra_state_slots)
        # set the has_tree_id marker to false
        self._state[-2] = 0
        assert int(self._state[-2]) == 0
        assert self._state[-2] == 0
        self.tree_depth = tree_depth

    def new_children(self, count, size=None):
        """Returns a list of new child nodes.

        Args:
            count (`int`): The number of child nodes to
                return.
            size (`int`): The state size to allocate for
                each child. The default value of None means
                the children will use the same state size as
                this node.
        """
        assert count >= 0
        if size is None:
            size = self.size
        tree_id = self.tree_id
        assert tree_id is not None
        bound = self.bound
        tree_depth = self.tree_depth
        children = []
        for i in range(count):
            children.append(self.__class__(size=size))
            self._insert_parent_tree_id(children[-1]._state, tree_id)
            children[-1].bound = bound
            children[-1].tree_depth = tree_depth + 1
            assert children[-1].tree_id is None
        return children

    @property
    def size(self):
        """Returns the size of the state storage array for this node"""
        return len(self._state) - self._extra_state_slots

    def resize(self, size):
        """Resizes the state storage array for this node"""
        orig_state = self._state
        self._state = self.new_array(size + self._extra_state_slots)
        self._state[-self._extra_state_slots:] = \
            orig_state[-self._extra_state_slots:]

    @property
    def state(self):
        """Returns the state storage array for this node"""
        return self._state[:self.size]

    @property
    def bound(self):
        """Get/set the bound stored with this node"""
        return self._extract_bound(self._state)
    @bound.setter
    def bound(self, bound):
        self._insert_bound(self._state, bound)

    @property
    def tree_id(self):
        """Get/set the tree id for this node (will be None
        if one has not been set)"""
        if self._has_tree_id(self._state):
            return self._extract_tree_id(self._state)
        return None
    @tree_id.setter
    def tree_id(self, tree_id):
        self._insert_tree_id(self._state, tree_id)

    @property
    def parent_tree_id(self):
        """Get the tree id of the parent of this node"""
        return self._extract_parent_tree_id(self._state)

    @property
    def tree_depth(self):
        """Get/set the tree depth for this node"""
        return self._extract_tree_depth(self._state)
    @tree_depth.setter
    def tree_depth(self, tree_depth):
        self._insert_tree_depth(self._state, tree_depth)

    #
    # class-level methods used by lower-level routines
    #

    @classmethod
    def _insert_best_objective(cls, state, best_objective):
        assert len(state) >= cls._extra_state_slots
        state[-6] = best_objective
        assert float(state[-6]) == float(best_objective)
        assert state[-6] == best_objective

    @classmethod
    def _extract_best_objective(cls, state):
        assert len(state) >= cls._extra_state_slots
        return float(state[-6])

    @classmethod
    def _insert_bound(cls, state, bound):
        assert len(state) >= cls._extra_state_slots
        state[-5] = bound
        assert float(state[-5]) == float(bound)
        assert state[-5] == bound

    @classmethod
    def _extract_bound(cls, state):
        assert len(state) >= cls._extra_state_slots
        return float(state[-5])

    @classmethod
    def _insert_tree_id(cls, state, tree_id):
        assert len(state) >= cls._extra_state_slots
        state[-4] = tree_id
        # make sure the floating point representation is
        # exact (tree_id is likely an integer)
        assert int(state[-4]) == int(tree_id)
        assert state[-4] == tree_id
        # set the has_tree_id marker to true
        state[-2] = 1
        assert int(state[-2]) == 1
        assert state[-2] == 1

    @classmethod
    def _extract_tree_id(cls, state):
        assert len(state) >= cls._extra_state_slots
        return int(state[-4])

    @classmethod
    def _insert_parent_tree_id(cls, state, tree_id):
        assert len(state) >= cls._extra_state_slots
        state[-3] = tree_id
        # make sure the floating point representation is
        # exact (tree_id is likely an integer)
        assert int(state[-3]) == int(tree_id)
        assert state[-3] == tree_id

    @classmethod
    def _extract_parent_tree_id(cls, state):
        assert len(state) >= cls._extra_state_slots
        return int(state[-3])

    @classmethod
    def _has_tree_id(cls, state):
        assert len(state) >= cls._extra_state_slots
        return int(state[-2]) == 1

    @classmethod
    def _insert_tree_depth(cls, state, tree_depth):
        assert len(state) >= cls._extra_state_slots
        state[-1] = tree_depth
        # make sure the floating point representation is
        # exact (tree_depth is likely an integer)
        assert int(state[-1]) == int(tree_depth)
        assert state[-1] == tree_depth

    @classmethod
    def _extract_tree_depth(cls, state):
        assert len(state) >= cls._extra_state_slots
        return int(state[-1])

class _ProblemBase(object):
    def __init__(self, sense):
        self.__sense = sense
        assert self.__sense in (minimize, maximize)

    @property
    def sense(self):
        """Returns the objective sense for this problem"""
        return self.__sense

    @property
    def infeasible_objective(self):
        """Returns the value representing an infeasible
        objective for this problem (i.e., +/-inf)"""
        if self.__sense == minimize:
            return infinity
        else:
            assert self.__sense == maximize
            return -infinity

    @property
    def unbounded_objective(self):
        """Returns the value representing an unbounded
        objective for this problem (i.e., +/-inf)"""
        if self.__sense == minimize:
            return -infinity
        else:
            assert self.__sense == maximize
            return infinity

    def compute_absolute_gap(self, bound, objective):
        """Returns the absolute gap between a bound and an
        objective, respecting the sign relative to the
        objective sense of this problem"""
        if bound == objective:
            return 0.0
        elif is_infinite(bound) or is_infinite(objective):
            if self.sense == minimize:
                if (bound == -infinity) or \
                   (objective == infinity):
                    return infinity
                else:
                    return -infinity
            else:
                assert self.sense == maximize
                if (bound == infinity) or \
                   (objective == -infinity):
                    return infinity
                else:
                    return -infinity
        else:
            if self.sense == minimize:
                gap = objective - bound
            else:
                assert self.sense == maximize
                gap = bound - objective
            return gap

    def compute_relative_gap(self, bound, objective):
        """Returns the relative gap between a bound and an
        objective, respecting the sign relative to the
        objective sense of this problem"""
        rgap = self.compute_absolute_gap(bound, objective)
        if is_infinite(rgap):
            return rgap
        rgap /= max(1.0, abs(objective))
        return rgap

class GenericProblem(_ProblemBase):

    def __init__(self,
                 sense,
                 absolute_gap=1e-9,
                 relative_gap=1e-6,
                 absolute_tolerance=1e-9,
                 cutoff=None):
        super(GenericProblem, self).__init__(sense)
        self._absolute_gap = float(absolute_gap)
        self._relative_gap = float(relative_gap)
        self._absolute_tolerance = float(absolute_tolerance)
        self._cutoff = None
        if cutoff is not None:
            self._cutoff = float(cutoff)
            assert not is_infinite(self._cutoff)

        assert (self._absolute_gap >= 0) and \
            (not is_infinite(self._absolute_gap))
        assert self._relative_gap >= 0 and \
            (not is_infinite(self._relative_gap))
        assert self._absolute_tolerance > 0 and \
            (not is_infinite(self._absolute_tolerance))

    @property
    def absolute_gap_tolerance(self):
        return self._absolute_gap

    @property
    def relative_gap_tolerance(self):
        return self._relative_gap

    def objective_is_optimal(self, objective, bound):
        assert bound != self.infeasible_objective
        if (objective != self.unbounded_objective) and \
           (objective != self.infeasible_objective):
            agap = self.compute_absolute_gap(bound,
                                             objective)
            if agap < self.absolute_gap_tolerance:
                return True
            else:
                rgap = self.compute_relative_gap(bound,
                                                 objective)
                if rgap < self.relative_gap_tolerance:
                    return True
        return False

    def bound_improved(self, new, old):
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return old > new + self._absolute_tolerance
        else:
            assert self.sense == maximize
            return old < new - self._absolute_tolerance

    def bound_worsened(self, new, old):
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return new < old - self._absolute_tolerance
        else:
            assert self.sense == maximize
            return new > old + self._absolute_tolerance

    def objective_improved(self, new, old):
        # handles the both equal and infinite case
        if old == new:
            return False
        if self.sense == minimize:
            return new < old - self._absolute_tolerance
        else:
            assert self.sense == maximize
            return new > old + self._absolute_tolerance

    def objective_can_improve(self, objective, bound):
        # handles the both equal and infinite case
        if bound == objective:
            return False
        if self.sense == minimize:
            return bound < objective - self._absolute_tolerance
        else:
            assert self.sense == maximize
            return bound > objective + self._absolute_tolerance

    def bound_is_suboptimal(self, bound, objective):
        # handles the both equal and infinite case
        if bound == objective:
            return False
        if self.sense == minimize:
            return bound > objective + self._absolute_tolerance
        else:
            assert self.sense == maximize
            return bound < objective - self._absolute_tolerance

    def cutoff_is_met(self, bound):
        if self._cutoff is not None:
            return self.bound_is_suboptimal(bound, self._cutoff)
        return False

class Problem(_ProblemBase):

    new_node = ProblemNode

    #
    # Abstract Methods
    #

    def objective(self):                          #pragma:nocover
        """Returns a feasible value for the objective of the
        current problem state or self.infeasible_objective
        if the current state is not feasible."""
        raise NotImplementedError()

    def bound(self):                              #pragma:nocover
        """Returns a value that is a bound on the objective
        of the current problem state."""
        raise NotImplementedError()

    def branch(self, parent):                     #pragma:nocover
        """Returns a list of
        :class:`pybnb.problem.ProblemNode` objects that
        partition the parent state into zero or more
        children."""
        raise NotImplementedError()

    def save_state(self, node):                   #pragma:nocover
        """Saves the current problem state into the given
        :class:`pybnb.problem.ProblemNode` object."""
        raise NotImplementedError()

    def load_state(self, node):                   #pragma:nocover
        """Loads the problem state that is stored on the
        given :class:`pybnb.problem.ProblemNode`
        object."""
        raise NotImplementedError()

    #
    # Optional Abstract Methods
    #

    def notify_new_best_objective_received(self,
                                           worker_comm,
                                           best_objective):
        """Called when a branch-and-bound solver receives a
        new best objective.

        Args:
            worker_comm: The worker MPI communicator. Will
                be None if MPI has been disabled.
            best_objective (`float`): The new best objective
                value.
        """
        pass

    def notify_new_best_objective(self,
                                  worker_comm,
                                  best_objective):
        """Called when a branch-and-bound solver locally
        computes a new best objective.

        Args:
            worker_comm: The worker MPI communicator. Will
                be None if MPI has been disabled.
            best_objective (`float`): The new best objective
                value.
        """
        pass

    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        """Called when a branch-and-bound solver finishes.

        Args:
            comm: The full MPI communicator that includes
                the dispatcher and all workers. Will be None
                if MPI has been disabled.
            worker_comm: The worker MPI communicator. Will
                be None if MPI has been disabled.
            results: The fully populated SolverResults
                container that will be returned from the
                solver.
        """
        pass
