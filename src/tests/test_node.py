from pybnb.node import Node

class TestNode(object):

    def test_best_objective_storage(self):
        node = Node()
        node._insert_best_objective(node._data, 10.5)
        assert node._extract_best_objective(node._data) == 10.5

    def test_children(self):
        parent = Node()
        assert parent.tree_id is None
        assert parent.tree_depth == 0
        assert len(parent.state) == 0
        parent.tree_id = 0
        parent.tree_depth = 1
        parent.bound = -1
        parent.resize(5)
        assert parent.tree_id == 0
        assert parent.tree_depth == 1
        assert parent.bound == -1
        assert len(parent.state) == 5
        children = parent.new_children(3)
        assert len(children) == 3
        for child in children:
            assert child.tree_id is None
            assert child.parent_tree_id == 0
            assert child.tree_depth == 2
            assert child.bound == -1
            assert len(child.state) == 5
        children = parent.new_children(4, size=10)
        assert len(children) == 4
        for child in children:
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
