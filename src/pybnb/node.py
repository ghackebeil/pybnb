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
    called to set size of the state array, which will be
    adjusted to the appropriate internal array size.

    Parameters
    ----------
    size : int, optional
        The size of the user portion of the internal storage
        array. (default: 0)
    tree_depth : int, optional
        The tree depth to store into the node. (default: 0)
    """
    __slots__ = ("_data", "_user_state")

    _extra_data_slots = 6
    # data[-6] = best_objective
    # data[-5] = bound
    # data[-4] = tree_id
    # data[-3] = parent_tree_id
    # data[-2] = has_tree_id
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
            # set the has_tree_id marker to false
            self._data[-2] = 0
            assert int(self._data[-2]) == 0
            assert self._data[-2] == 0
            self.tree_depth = tree_depth

    def _set_data(self, data):
        assert len(data) >= self._extra_data_slots
        self._data = data
        self._user_state = self._data[:-self._extra_data_slots]

    def new_children(self, count, size=None):
        """Returns a list of new child nodes.

        Parameters
        ----------
        count : int
            The number of child nodes to return.
        size : int, optional
            The state size to allocate for each child. If
            set to None, the children will use the same
            state size as this node. (default: None)
        """
        assert count >= 0
        if size is None:
            size = len(self._user_state)
        tree_id = self.tree_id
        assert tree_id is not None
        bound = self.bound
        tree_depth = self.tree_depth
        children = []
        for i in range(count):
            children.append(Node(size=size))
            self._insert_parent_tree_id(children[-1]._data, tree_id)
            children[-1].bound = bound
            children[-1].tree_depth = tree_depth + 1
            assert children[-1].tree_id is None
        return children

    def resize(self, size, force_new=False):
        """Resize the state storage array for this node.

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
        """
        assert size >= 0
        if (len(self._data) != size + self._extra_data_slots) or \
           force_new:
            orig_data = self._data
            orig_user_state = self._user_state
            self._set_data(numpy.empty(size + self._extra_data_slots,
                                       dtype=float))
            # both _data and _user_state are updated
            # at this point
            self._data[-self._extra_data_slots:] = \
                orig_data[-self._extra_data_slots:]
            min_size = min(len(self._user_state),
                           len(orig_user_state))
            self._user_state[:min_size] = orig_user_state[:min_size]

    @property
    def state(self):
        """Returns the user state storage array for this node."""
        return self._user_state

    @property
    def bound(self):
        """Get/set the bound for this node."""
        return self._extract_bound(self._data)
    @bound.setter
    def bound(self, bound):
        self._insert_bound(self._data, bound)

    @property
    def tree_id(self):
        """Get/set the tree id for this node. This defaults
        to None when a node is created."""
        if self._has_tree_id(self._data):
            return self._extract_tree_id(self._data)
        return None
    @tree_id.setter
    def tree_id(self, tree_id):
        self._insert_tree_id(self._data, tree_id)

    @property
    def parent_tree_id(self):
        """Get the tree id of the parent for this node. This
        attribute will be automatically set on nodes returned
        from the :func:`pybnb.node.Node.new_children`
        method."""
        return self._extract_parent_tree_id(self._data)

    @property
    def tree_depth(self):
        """Get/set the tree depth for this node. This
        attribute will be automatically set on nodes
        returned from the
        :func:`pybnb.node.Node.new_children` method (to 1
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
        data[-6] = best_objective
        assert float(data[-6]) == float(best_objective)
        assert data[-6] == best_objective

    @classmethod
    def _extract_best_objective(cls, data):
        assert len(data) >= cls._extra_data_slots
        return float(data[-6])

    @classmethod
    def _insert_bound(cls, data, bound):
        assert len(data) >= cls._extra_data_slots
        data[-5] = bound
        assert float(data[-5]) == float(bound)
        assert data[-5] == bound

    @classmethod
    def _extract_bound(cls, data):
        assert len(data) >= cls._extra_data_slots
        return float(data[-5])

    @classmethod
    def _insert_tree_id(cls, data, tree_id):
        assert len(data) >= cls._extra_data_slots
        data[-4] = tree_id
        # make sure the floating point representation is
        # exact (tree_id is likely an integer)
        assert int(data[-4]) == int(tree_id)
        assert data[-4] == tree_id
        # set the has_tree_id marker to true
        data[-2] = 1
        assert int(data[-2]) == 1
        assert data[-2] == 1

    @classmethod
    def _extract_tree_id(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-4])

    @classmethod
    def _insert_parent_tree_id(cls, data, tree_id):
        assert len(data) >= cls._extra_data_slots
        data[-3] = tree_id
        # make sure the floating point representation is
        # exact (tree_id is likely an integer)
        assert int(data[-3]) == int(tree_id)
        assert data[-3] == tree_id

    @classmethod
    def _extract_parent_tree_id(cls, data):
        assert len(data) >= cls._extra_data_slots
        return int(data[-3])

    @classmethod
    def _has_tree_id(cls, data):
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
