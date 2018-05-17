import pytest

from pybnb.node import Node

class TestNode(object):

    def test_best_objective_storage(self):
        node = Node()
        node._insert_best_objective(node._data, 10.5)
        assert node._extract_best_objective(node._data) == 10.5

    def test_children(self):
        parent = Node()
        assert parent.queue_priority is None
        assert parent.tree_id is None
        assert parent.parent_tree_id is None
        assert parent.tree_depth == 0
        assert len(parent.state) == 0
        parent.queue_priority = 10
        assert parent.queue_priority == 10
        assert parent.tree_id is None
        assert parent.parent_tree_id is None
        assert parent.tree_depth == 0
        assert len(parent.state) == 0
        parent.tree_depth = 1
        assert parent.queue_priority == 10
        assert parent.tree_id is None
        assert parent.parent_tree_id is None
        assert parent.tree_depth == 1
        assert len(parent.state) == 0
        parent.bound = -1
        assert parent.queue_priority == 10
        assert parent.tree_id is None
        assert parent.parent_tree_id is None
        assert parent.tree_depth == 1
        assert parent.bound == -1
        assert len(parent.state) == 0
        parent.resize(5)
        assert parent.queue_priority == 10
        assert parent.tree_id is None
        assert parent.parent_tree_id is None
        assert parent.tree_depth == 1
        assert parent.bound == -1
        assert len(parent.state) == 5
        children = [parent.new_child()
                    for i in range(3)]
        assert len(children) == 3
        for child in children:
            assert child.queue_priority is None
            assert child.tree_id is None
            assert child.parent_tree_id is None
            assert child.tree_depth == 2
            assert child.bound == -1
            assert len(child.state) == 5
        Node._insert_tree_id(parent._data, 0)
        assert parent.tree_id == 0
        children = [parent.new_child()
                    for i in range(3)]
        assert len(children) == 3
        for child in children:
            assert child.queue_priority is None
            assert child.tree_id is None
            assert child.parent_tree_id == 0
            assert child.tree_depth == 2
            assert child.bound == -1
            assert len(child.state) == 5
        children = [parent.new_child(size=10)
                    for i in range(4)]
        assert len(children) == 4
        for child in children:
            assert child.queue_priority is None
            assert child.tree_id is None
            assert child.parent_tree_id == 0
            assert child.tree_depth == 2
            assert child.bound == -1
            assert len(child.state) == 10

    def test_state_update(self):
        node = Node(size=3)
        node.state[0] = 1.1
        node.state[1] = 0.0
        node.state[2] = 0.0
        assert node.state[0] == 1.1
        assert node.state[1] == 0
        assert node.state[2] == 0
        node.state[1:3] = [-1.0, 5.3]
        assert node.state[0] == 1.1
        assert node.state[1] == -1.0
        assert node.state[2] == 5.3

        state = node.state
        node.resize(4)
        with pytest.raises(ValueError):
            state[0] = 1
