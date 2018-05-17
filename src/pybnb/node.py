"""
Branch-and-bound node implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import numpy

class Node(object):
    """A branch-and-bound node that stores problem state
    information inside of a NumPy array (dtype=float).

    This class maintains an internal storage array and
    exposes a portion of that array to the user through the
    :attr:`state <pybnb.node.Node.state>` attribute.  The
    :func:`resize <pybnb.node.Node.resize>` method should be
    called to set the amount of available user state
    storage, which will be adjusted to the appropriate
    internal array size.

    Parameters
    ----------
    size : int, optional
        The size of the user portion of the internal storage
        array. (default: 0)
    tree_depth : int, optional
        The tree depth to store into the node. (default: 0)
    """
    __slots__ = ("_data", "_user_state")

    _extra_data_slots = 9
    # data[-9] = best_objective
    # data[-8] = bound
    # data[-7] = queue_priority
    # data[-6] = has_queue_priority
    # data[-5] = tree_id
    # data[-4] = has_tree_id
    # data[-3] = parent_tree_id
    # data[-2] = has_parent_tree_id
    # data[-1] = tree_depth

    def __init__(self, size=0, tree_depth=0, data_=None):
        self._data = None
        self._user_state = None
        if data_ is not None:
            assert size == 0
            assert tree_depth == 0
            self._set_data(data_)
        else:
            assert size >= 0
            assert tree_depth >= 0
            self._set_data(numpy.empty(size + self._extra_data_slots,
                                       dtype=float))
            # set the has_queue_priority marker to false
            self._data[-6] = 0
            assert int(self._data[-6]) == 0
            assert self._data[-6] == 0
            # set the has_tree_id marker to false
            self._data[-4] = 0
            assert int(self._data[-4]) == 0
            assert self._data[-4] == 0
            # set the has_parent_tree_id marker to false
            self._data[-2] = 0
            assert int(self._data[-2]) == 0
            assert self._data[-2] == 0
            # set the tree depth
            self.tree_depth = tree_depth

    def _set_data(self, data):
        assert len(data) >= self._extra_data_slots
        assert type(data) is numpy.ndarray
        self._data = data
        self._user_state = self._data[:-self._extra_data_slots]
        # make _user_state is a view of _data
        if self._data.base is None:
            assert self._user_state.base is self._data
        else:
            assert self._user_state.base is self._data.base

    def new_child(self, size=None):
        """Returns a new child node.

        Parameters
        ----------
        size : int, optional
            The state size to allocate for each child. If
            set to None, the children will use the same
            state size as this node. (default: None)
        """
        if size is None:
            size = len(self._user_state)
        tree_id = self.tree_id
        bound = self.bound
        tree_depth = self.tree_depth
        child = Node(size=size)
        if tree_id is not None:
            self._insert_parent_tree_id(child._data, tree_id)
        else:
            # set the has_parent_tree_id flag to False on the child
            child._data[-2] = 0
        child.bound = bound
        child.tree_depth = tree_depth + 1
        assert child.tree_id is None
        return child

    def resize(self, size, force_new=False):
        """Resize the user state storage array for this node.

        Parameters
        ----------
        size : int
            The number of entries to allocate for the public
            state array.
        force_new : bool
            Indicate whether or not a new array should be
            created even if the size does not change.  The
            default behavior will only reallocate a new
            array when the size changes. (default: False)

        Note
        ----
            If a new array is allocated, the previous user
            state array (accessible via the :attr:`state
            <pybnb.node.Node.state>` attribute), will become
            readonly as it will be invalidated.
        """
        assert size >= 0
        if (len(self._data) != size + self._extra_data_slots) or \
           force_new:
            orig_data = self._data
            orig_user_state = self._user_state
            # make the user state view for the previous data
            # array readonly
            orig_user_state.setflags(write=False)
            self._set_data(numpy.empty(size + self._extra_data_slots,
                                       dtype=float))
            # both _data and _user_state are updated at this
            # point, now copy the original values where
            # indices match
            self._data[-self._extra_data_slots:] = \
                orig_data[-self._extra_data_slots:]
            min_size = min(len(self._user_state),
                           len(orig_user_state))
            self._user_state[:min_size] = orig_user_state[:min_size]

    @property
    def state(self):
        """Returns the user state numeric storage array for
        this node. This will be a view on a larger numpy
        array created with ``dtype=float``.

        Note
        ----
            The :func:`resize <pybnb.node.Node.resize>`
            method is provided to allow users to change the
            amount of storage available in this array. If
            the previous user state view becomes invalid due
            to a resize, that array view will be marked as
            readonly.

        Example
        -------

        >>> node = Node(size=1)
        >>> assert len(node.state) == 1
        >>> state = node.state
        >>> state[0] = 0
        >>> # now resize (causing state to become a stale view)
        >>> node.resize(2)
        >>> state[0] = 1
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ValueError: assignment destination is read-only
        >>> node.state[1] = 1
        >>> assert list(node.state) == [0,1]

        """
        return self._user_state

    @property
    def queue_priority(self):
        """Get/set the queue priority for this node."""
        if self._has_queue_priority(self._data):
            return self._extract_queue_priority(self._data)
        return None
    @queue_priority.setter
    def queue_priority(self, queue_priority):
        self._insert_queue_priority(self._data, queue_priority)

    @property
    def bound(self):
        """Get/set the bound for this node."""
        return self._extract_bound(self._data)
    @bound.setter
    def bound(self, bound):
        self._insert_bound(self._data, bound)

    @property
    def tree_id(self):
        """Get the tree id for this node. This defaults to
        None when a node is created and will be set by the
        dispatcher when the node is added to the work
        queue."""
        if self._has_tree_id(self._data):
            return self._extract_tree_id(self._data)
        return None

    @property
    def parent_tree_id(self):
        """Get the tree id of the parent for this node. This
        attribute will be automatically set on nodes returned
        from the :func:`pybnb.node.Node.new_child`
        method."""
        if self._has_parent_tree_id(self._data):
            return self._extract_parent_tree_id(self._data)
        return None

    @property
    def tree_depth(self):
        """Get/set the tree depth for this node. This
        attribute will be automatically set on nodes
        returned from the
        :func:`pybnb.node.Node.new_child` method (to 1
        more than the value stored on this node)."""
        return self._extract_tree_depth(self._data)
    @tree_depth.setter
    def tree_depth(self, tree_depth):
        self._insert_tree_depth(self._data, tree_depth)

    #
    # class-level methods used by lower-level routines
    #

    @classmethod
    def _insert_best_objective(cls, data, best_objective):
        assert len(data) >= cls._extra_data_slots
        data[-9] = best_objective
        assert float(data[-9]) == float(best_objective)
        assert data[-9] == best_objective

    @classmethod
    def _extract_best_objective(cls, data):
        assert len(data) >= cls._extra_data_slots
        return float(data[-9])

    @classmethod
    def _insert_bound(cls, data, bound):
        assert len(data) >= cls._extra_data_slots
        data[-8] = bound
        assert float(data[-8]) == float(bound)
        assert data[-8] == bound

    @classmethod
    def _extract_bound(cls, data):
        assert len(data) >= cls._extra_data_slots
        return float(data[-8])

    @classmethod
    def _insert_queue_priority(cls, data, queue_priority):
        assert len(data) >= cls._extra_data_slots
        data[-7] = queue_priority
        assert data[-7] == queue_priority
        # set the has_queue_priority marker to true
        data[-6] = 1
        assert int(data[-6]) == 1
        assert data[-6] == 1

    @classmethod
    def _extract_queue_priority(cls, data):
        assert len(data) >= cls._extra_data_slots
        return float(data[-7])

    @classmethod
    def _has_queue_priority(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-6]) == 1

    @classmethod
    def _insert_tree_id(cls, data, tree_id):
        assert len(data) >= cls._extra_data_slots
        data[-5] = tree_id
        # make sure the floating point representation is
        # exact (tree_id is likely an integer)
        assert int(data[-5]) == int(tree_id)
        assert data[-5] == tree_id
        # set the has_tree_id marker to true
        data[-4] = 1
        assert int(data[-4]) == 1
        assert data[-4] == 1

    @classmethod
    def _extract_tree_id(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-5])

    @classmethod
    def _has_tree_id(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-4]) == 1

    @classmethod
    def _insert_parent_tree_id(cls, data, tree_id):
        assert len(data) >= cls._extra_data_slots
        data[-3] = tree_id
        # make sure the floating point representation is
        # exact (tree_id is likely an integer)
        assert int(data[-3]) == int(tree_id)
        assert data[-3] == tree_id
        # set the has_tree_id marker to true
        data[-2] = 1
        assert int(data[-2]) == 1
        assert data[-2] == 1

    @classmethod
    def _extract_parent_tree_id(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-3])

    @classmethod
    def _has_parent_tree_id(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-2]) == 1

    @classmethod
    def _insert_tree_depth(cls, data, tree_depth):
        assert len(data) >= cls._extra_data_slots
        data[-1] = tree_depth
        # make sure the floating point representation is
        # exact (tree_depth is likely an integer)
        assert int(data[-1]) == int(tree_depth)
        assert data[-1] == tree_depth

    @classmethod
    def _extract_tree_depth(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-1])
