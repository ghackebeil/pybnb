"""
Branch-and-bound node implementation.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import uuid
import zlib

from pybnb.configuration import config

import six

if not six.PY2:
    import pickle
else:
    import cPickle as pickle

_serializer_modules = {}
_serializer_modules["pickle"] = pickle
_serializer_modules["dill"] = None

def _get_dill():
    assert config.SERIALIZER == "dill"
    import dill
    _serializer_modules["dill"] = dill
    return dill

def dumps(obj):
    """Return the serialized representation of the object as
    a bytes object, using the serialization module set in
    the current configuration."""
    try:
        mod = _serializer_modules[config.SERIALIZER]
    except KeyError:
        raise ValueError("Invalid serializer '%s'. "
                         "Valid choices are: ['pickle', 'dill']"
                         % (config.SERIALIZER))
    if mod is None:
        mod = _get_dill()
    data = mod.dumps(
        obj,
        protocol=config.SERIALIZER_PROTOCOL_VERSION)
    if config.COMPRESSION:
        data = zlib.compress(data)
    return data

def loads(data):
    """Read and return an object from the given serialized
    data, using the serialization module set in the current
    configuration."""
    try:
        mod = _serializer_modules[config.SERIALIZER]
    except KeyError:
        raise ValueError("Invalid serializer '%s'. "
                         "Valid choices are: ['pickle', 'dill']"
                         % (config.SERIALIZER))
    if mod is None:
        mod = _get_dill()
    if config.COMPRESSION:
        data = zlib.decompress(data)
    return mod.loads(data)

class _SerializedNode(object):
    """A helper object used by the distributed dispatcher
    for lightweight handling of serialized nodes."""
    __slots__ = ("objective",
                 "bound",
                 "tree_depth",
                 "queue_priority",
                 "_uuid",
                 "data")
    def __init__(self, slots):
        (self.objective,
         self.bound,
         self.tree_depth,
         self.queue_priority,
         self._uuid,
         self.data) = slots

    def _generate_uuid(self):
        self._uuid = uuid.uuid4().hex

    @property
    def slots(self):
        return (self.objective,
                self.bound,
                self.tree_depth,
                self.queue_priority,
                self._uuid,
                self.data)

    @staticmethod
    def to_slots(node):
        return (node.objective,
                node.bound,
                node.tree_depth,
                node.queue_priority,
                node._uuid,
                dumps(node.state))

    @classmethod
    def from_node(cls, node):
        return cls(cls.to_slots(node))

    @staticmethod
    def restore_node(slots):
        node = Node()
        (node.objective,
         node.bound,
         node.tree_depth,
         node.queue_priority,
         node._uuid,
         node.state) = slots
        node.state = loads(node.state)
        return node

class Node(object):
    """A branch-and-bound node that stores problem state.

    Attributes
    ----------
    objective : float
        The objective value for the node.
    bound : float
        The bound value for the node.
    tree_depth : int
        The tree depth of the node (0-based).
    queue_priority : float or tuple of floats
        The queue priority of the node.
    state
        The user specified node state.
    """
    __slots__ = ("objective",
                 "bound",
                 "tree_depth",
                 "queue_priority",
                 "_uuid",
                 "state")

    def __init__(self):
        self.objective = None
        self.bound = None
        self.tree_depth = None
        self.queue_priority = None
        self._uuid = None
        self.state = None

    def _generate_uuid(self):
        self._uuid = uuid.uuid4().hex

    def resize(self, *args, **kwds):
        raise NotImplementedError(
            "It is no longer necessary to call "
            "node.resize(...). Simply assign any object "
            "to the node.state attribute. It no longer "
            "needs to be a numpy array. The node state "
            "must be serialize-able using pickle or dill "
            "in order to be compatible with the MPI-based "
            "parallel solver.")

    def __str__(self):
        out = \
            ("Node(objective=%s,\n"
             "     bound=%s,\n"
             "     tree_depth=%s)"
             % (self.objective,
                self.bound,
                self.tree_depth))
        return out

    def new_child(self):
        child = Node()
        child.objective = self.objective
        child.bound = self.bound
        child.tree_depth = self.tree_depth + 1
        assert child.queue_priority is None
        assert child.state is None
        return child
