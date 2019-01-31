import pytest

from pybnb.configuration import config
from pybnb.node import (loads,
                        dumps,
                        _SerializedNode,
                        Node)

class TestNode(object):

    def test_bad_serializer(self):
        orig = config.SERIALIZER
        config.SERIALIZER = "_not_dill_or_pickle_"
        try:
            with pytest.raises(ValueError):
                loads(b'')
            with pytest.raises(ValueError):
                dumps(None)
        finally:
            config.SERIALIZER = orig
        print(config)

    def test_resize(self):
        node = Node()
        with pytest.raises(NotImplementedError):
            node.resize(1)

    def test_init(self):
        node = Node()
        assert node.objective is None
        assert node.bound is None
        assert node.tree_id is None
        assert node.parent_tree_id is None
        assert node.tree_depth == 0
        assert node.queue_priority is None
        assert node.state is None

    def test_new_child(self):
        node = Node()
        assert node.objective is None
        assert node.bound is None
        assert node.tree_id is None
        assert node.parent_tree_id is None
        assert node.tree_depth == 0
        assert node.queue_priority is None
        assert node.state is None
        node = node.new_child()
        assert node.objective is None
        assert node.bound is None
        assert node.tree_id is None
        assert node.parent_tree_id is None
        assert node.tree_depth == 1
        assert node.queue_priority is None
        assert node.state is None
        node.objective = 1
        node.bound = -1
        node.tree_id = 0
        node.queue_priority = 5
        node.state = 'a'
        node = node.new_child()
        assert node.objective == 1
        assert node.bound == -1
        assert node.tree_id is None
        assert node.parent_tree_id == 0
        assert node.tree_depth == 2
        assert node.queue_priority is None
        assert node.state is None

    def test_str(self):
        node = Node()
        node.objective = -1
        node.bound = -2
        node.tree_id = 1
        node.parent_tree_id = 0
        node.tree_depth = 3
        node.queue_priority = (1,2,3)
        node.state = 'a'
        assert str(node) == \
            """\
Node(objective=-1,
     bound=-2,
     tree_id=1,
     parent_tree_id=0,
     tree_depth=3,
     queue_priority=(1, 2, 3))"""

    def test_serialization(self):
        node = Node()
        node.objective = 0.0
        node.bound = 1.0
        node.tree_id = 1
        node.parent_tree_id = 0
        node.tree_depth = -1
        node.queue_priority = (1,2,3)
        node.state = 'a'
        s = _SerializedNode.from_node(node)
        assert s.objective == node.objective
        assert s.bound == node.bound
        assert s.tree_id == node.tree_id
        assert s.parent_tree_id == node.parent_tree_id
        assert s.tree_depth == node.tree_depth
        assert s.queue_priority == node.queue_priority
        assert s.data is not None
        node_ = s.restore_node(s.slots)
        assert node_.objective == node.objective
        assert node_.bound == node.bound
        assert node_.tree_id == node.tree_id
        assert node_.parent_tree_id == node.parent_tree_id
        assert node_.tree_depth == node.tree_depth
        assert node_.queue_priority == node.queue_priority
        assert node_.state == node.state
