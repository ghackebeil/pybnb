"""
Branch-and-bound node implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import numpy

class Node(object):
    """A branch-and-bound node used to store problem state
    information."""
    __slots__ = ("_state",)

    _extra_state_slots = 6
    # state[-6] = best_objective
    # state[-5] = bound
    # state[-4] = tree_id
    # state[-3] = parent_tree_id
    # state[-2] = has_tree_id
    # state[-1] = tree_depth

    def __init__(self, size=0, tree_depth=0):
        assert size >= 0
        assert tree_depth >= 0
        self._state = numpy.empty(size + self._extra_state_slots,
                                  dtype=float)
        # set the has_tree_id marker to false
        self._state[-2] = 0
        assert int(self._state[-2]) == 0
        assert self._state[-2] == 0
        self.tree_depth = tree_depth

    def new_children(self, count, size=None):
        """Returns a list of new child nodes.

        Parameters
        ----------
        count : int
            The number of child nodes to return.
        size : int, optional
            The state size to allocate for each child. If
            set to None, the children will use the same
            state size as this node. (default=None)
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
            children.append(Node(size=size))
            self._insert_parent_tree_id(children[-1]._state, tree_id)
            children[-1].bound = bound
            children[-1].tree_depth = tree_depth + 1
            assert children[-1].tree_id is None
        return children

    @property
    def size(self):
        """Returns the size of the state storage array for
        this node."""
        return len(self._state) - self._extra_state_slots

    def resize(self, size):
        """Resizes the state storage array for this node."""
        orig_state = self._state
        self._state = numpy.empty(size + self._extra_state_slots,
                                  dtype=float)
        self._state[-self._extra_state_slots:] = \
            orig_state[-self._extra_state_slots:]

    @property
    def state(self):
        """Returns the state storage array for this node."""
        return self._state[:self.size]

    @property
    def bound(self):
        """Get/set the bound stored with this node."""
        return self._extract_bound(self._state)
    @bound.setter
    def bound(self, bound):
        self._insert_bound(self._state, bound)

    @property
    def tree_id(self):
        """Get/set the tree id for this node (will be None
        if one has not been set)."""
        if self._has_tree_id(self._state):
            return self._extract_tree_id(self._state)
        return None
    @tree_id.setter
    def tree_id(self, tree_id):
        self._insert_tree_id(self._state, tree_id)

    @property
    def parent_tree_id(self):
        """Get the tree id of the parent of this node."""
        return self._extract_parent_tree_id(self._state)

    @property
    def tree_depth(self):
        """Get/set the tree depth for this node."""
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
