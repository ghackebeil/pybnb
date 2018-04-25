import itertools

import pytest

from pybnb.misc import (infinity,
                        is_infinite)
from pybnb.problem import (minimize,
                           maximize,
                           ProblemNode,
                           GenericProblem,
                           Problem)

class TestProblemNode(object):

    def test_best_objective_storage(self):
        node = ProblemNode()
        node._insert_best_objective(node._state, 10.5)
        assert node._extract_best_objective(node._state) == 10.5

    def test_size(self):
        node = ProblemNode()
        assert node.size == 0
        node = ProblemNode(size=2)
        assert node.size == 2

    def test_children(self):
        parent = ProblemNode()
        assert parent.tree_id is None
        assert parent.tree_depth == 0
        assert parent.size == 0
        assert len(parent.state) == 0
        parent.tree_id = 0
        parent.tree_depth = 1
        parent.bound = -1
        parent.resize(5)
        assert parent.tree_id == 0
        assert parent.tree_depth == 1
        assert parent.bound == -1
        assert parent.size == 5
        assert len(parent.state) == 5
        children = parent.new_children(3)
        assert len(children) == 3
        for child in children:
            assert child.tree_id is None
            assert child.parent_tree_id == 0
            assert child.tree_depth == 2
            assert child.bound == -1
            assert child.size == 5
            assert len(child.state) == 5
        children = parent.new_children(4, size=10)
        assert len(children) == 4
        for child in children:
            assert child.tree_id is None
            assert child.parent_tree_id == 0
            assert child.tree_depth == 2
            assert child.bound == -1
            assert child.size == 10
            assert len(child.state) == 10

    def test_state_update(self):
        node = ProblemNode(size=3)
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

class TestGenericProblem(object):

    def test_sense(self):
        p = GenericProblem(minimize)
        assert p.sense == minimize
        p = GenericProblem(maximize)
        assert p.sense == maximize

    def test_infeasible_objective(self):
        p = GenericProblem(minimize)
        assert p.infeasible_objective == infinity
        p = GenericProblem(maximize)
        assert p.infeasible_objective == -infinity

    def test_unbounded_objective(self):
        p = GenericProblem(minimize)
        assert p.unbounded_objective == -infinity
        p = GenericProblem(maximize)
        assert p.unbounded_objective == infinity

    def test_objective_is_optimal(self):

        for sense in [minimize, maximize]:
            p = GenericProblem(sense)
            for bound,objective in itertools.product([-infinity,
                                                      infinity,
                                                      0.0],
                                                     [-infinity,
                                                      infinity,
                                                      0.0]):
                if is_infinite(bound) and is_infinite(objective):
                    if bound != p.infeasible_objective:
                        assert not p.objective_is_optimal(objective, bound)
                elif objective == bound:
                    assert p.objective_is_optimal(objective, bound)
                elif bound != p.infeasible_objective:
                    assert not p.objective_is_optimal(objective, bound)

    def test_compute_relative_gap(self):

        for sense in [minimize, maximize]:
            p = GenericProblem(sense)
            for bound,objective in itertools.product([-infinity,
                                                      infinity,
                                                      0.0],
                                                     [-infinity,
                                                      infinity,
                                                      0.0]):
                if (not is_infinite(bound)) or \
                   (not is_infinite(objective)):
                    continue
                if bound == objective:
                    assert p.compute_relative_gap(bound,objective) == 0
                elif p.sense == minimize:
                    if (bound == -infinity) or \
                       (objective == infinity):
                        assert p.compute_relative_gap(bound,objective) == \
                            infinity
                    else:
                        assert p.compute_relative_gap(bound,objective) == \
                            -infinity
                else:
                    assert p.sense == maximize
                    if (bound == infinity) or \
                       (objective == -infinity):
                        assert p.compute_relative_gap(bound,objective) == \
                            infinity
                    else:
                        assert p.compute_relative_gap(bound,objective) == \
                            -infinity

    def test_bound_improved(self):
        # min
        p = GenericProblem(minimize)
        assert not p.bound_improved(-infinity,
                                    -infinity)
        assert p.bound_improved(-infinity,
                                infinity)
        assert p.bound_improved(-infinity,
                                0.0)
        assert not p.bound_improved(infinity,
                                    -infinity)
        assert not p.bound_improved(infinity,
                                    infinity)
        assert not p.bound_improved(infinity,
                                    0.0)
        assert not p.bound_improved(0.0,
                                    -infinity)
        assert p.bound_improved(0.0,
                                infinity)
        assert not p.bound_improved(0.0, 0.0)
        # max
        p = GenericProblem(maximize)
        assert not p.bound_improved(-infinity,
                                    -infinity)
        assert not p.bound_improved(-infinity,
                                    infinity)
        assert not p.bound_improved(-infinity,
                                    0.0)
        assert p.bound_improved(infinity,
                                -infinity)
        assert not p.bound_improved(infinity,
                                    infinity)
        assert p.bound_improved(infinity,
                                0.0)
        assert p.bound_improved(0.0,
                                -infinity)
        assert not p.bound_improved(0.0,
                                    infinity)
        assert not p.bound_improved(0.0,
                                    0.0)

    def test_bound_worsened(self):
        # min
        p = GenericProblem(minimize)
        assert not p.bound_worsened(-infinity,
                                -infinity)
        assert p.bound_worsened(-infinity,
                                infinity)
        assert p.bound_worsened(-infinity,
                                0.0)
        assert not p.bound_worsened(infinity,
                                    -infinity)
        assert not p.bound_worsened(infinity,
                                    infinity)
        assert not p.bound_worsened(infinity,
                                    0.0)
        assert not p.bound_worsened(0.0,
                                    -infinity)
        assert p.bound_worsened(0.0,
                                infinity)
        assert not p.bound_worsened(0.0,
                                    0.0)
        # max
        p = GenericProblem(maximize)
        assert not p.bound_worsened(-infinity,
                                    -infinity)
        assert not p.bound_worsened(-infinity,
                                    infinity)
        assert not p.bound_worsened(-infinity,
                                    0.0)
        assert p.bound_worsened(infinity,
                                -infinity)
        assert not p.bound_worsened(infinity,
                                    infinity)
        assert p.bound_worsened(infinity,
                                0.0)
        assert p.bound_worsened(0.0,
                                -infinity)
        assert not p.bound_worsened(0.0,
                                    infinity)
        assert not p.bound_worsened(0.0,
                                    0.0)

    def test_objective_improved(self):
        # min
        p = GenericProblem(minimize)
        assert not p.objective_improved(-infinity,
                                        -infinity)
        assert p.objective_improved(-infinity,
                                    infinity)
        assert p.objective_improved(-infinity,
                                    0.0)
        assert not p.objective_improved(infinity,
                                        -infinity)
        assert not p.objective_improved(infinity,
                                        infinity)
        assert not p.objective_improved(infinity,
                                        0.0)
        assert not p.objective_improved(0.0,
                                        -infinity)
        assert p.objective_improved(0.0,
                                    infinity)
        assert not p.objective_improved(0.0,
                                        0.0)
        # max
        p = GenericProblem(maximize)
        assert not p.objective_improved(-infinity,
                                        -infinity)
        assert not p.objective_improved(-infinity,
                                        infinity)
        assert not p.objective_improved(-infinity,
                                        0.0)
        assert p.objective_improved(infinity,
                                    -infinity)
        assert not p.objective_improved(infinity,
                                        infinity)
        assert p.objective_improved(infinity,
                                    0.0)
        assert p.objective_improved(0.0,
                                    -infinity)
        assert not p.objective_improved(0.0,
                                        infinity)
        assert not p.objective_improved(0.0,
                                        0.0)

    def test_objective_can_improve(self):
        # min
        p = GenericProblem(minimize)
        assert not p.objective_can_improve(-infinity,
                                           -infinity)
        assert not p.objective_can_improve(-infinity,
                                           infinity)
        assert not p.objective_can_improve(-infinity,
                                           0.0)
        assert p.objective_can_improve(infinity,
                                       -infinity)
        assert not p.objective_can_improve(infinity,
                                           infinity)
        assert p.objective_can_improve(infinity,
                                       0.0)
        assert p.objective_can_improve(0.0,
                                       -infinity)
        assert not p.objective_can_improve(0.0,
                                           infinity)
        assert not p.objective_can_improve(0.0,
                                           0.0)
        # max
        p = GenericProblem(maximize)
        assert not p.objective_can_improve(-infinity,
                                           -infinity)
        assert p.objective_can_improve(-infinity,
                                       infinity)
        assert p.objective_can_improve(-infinity,
                                       0.0)
        assert not p.objective_can_improve(infinity,
                                           -infinity)
        assert not p.objective_can_improve(infinity,
                                           infinity)
        assert not p.objective_can_improve(infinity,
                                           0.0)
        assert not p.objective_can_improve(0.0,
                                           -infinity)
        assert p.objective_can_improve(0.0,
                                       infinity)
        assert not p.objective_can_improve(0.0,
                                           0.0)

    def test_bound_is_suboptimal(self):
        # min
        p = GenericProblem(minimize)
        assert not p.bound_is_suboptimal(-infinity,
                                         -infinity)
        assert not p.bound_is_suboptimal(-infinity,
                                         infinity)
        assert not p.bound_is_suboptimal(-infinity,
                                         0.0)
        assert p.bound_is_suboptimal(infinity,
                                     -infinity)
        assert not p.bound_is_suboptimal(infinity,
                                         infinity)
        assert p.bound_is_suboptimal(infinity,
                                     0.0)
        assert p.bound_is_suboptimal(0.0,
                                     -infinity)
        assert not p.bound_is_suboptimal(0.0,
                                         infinity)
        assert not p.bound_is_suboptimal(0.0,
                                         0.0)
        # max
        p = GenericProblem(maximize)
        assert not p.bound_is_suboptimal(-infinity,
                                         -infinity)
        assert p.bound_is_suboptimal(-infinity,
                                     infinity)
        assert p.bound_is_suboptimal(-infinity,
                                     0.0)
        assert not p.bound_is_suboptimal(infinity,
                                         -infinity)
        assert not p.bound_is_suboptimal(infinity,
                                         infinity)
        assert not p.bound_is_suboptimal(infinity,
                                         0.0)
        assert not p.bound_is_suboptimal(0.0,
                                         -infinity)
        assert p.bound_is_suboptimal(0.0,
                                     infinity)
        assert not p.bound_is_suboptimal(0.0,
                                         0.0)

class TestProblem(object):

    def test_sense(self):
        p = Problem(minimize)
        assert p.sense == minimize
        p = Problem(maximize)
        assert p.sense == maximize

    def test_infeasible_objective(self):
        p = Problem(minimize)
        assert p.infeasible_objective == infinity
        p = Problem(maximize)
        assert p.infeasible_objective == -infinity

    def test_unbounded_objective(self):
        p = Problem(minimize)
        assert p.unbounded_objective == -infinity
        p = Problem(maximize)
        assert p.unbounded_objective == infinity
