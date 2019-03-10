import pytest

from pybnb.common import (minimize,
                          maximize,
                          inf)
from pybnb.misc import get_simple_logger
from pybnb.node import Node
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.dispatcher import (DispatcherLocal,
                              DispatcherQueueData)
from pybnb.priority_queue import (WorstBoundFirstPriorityQueue,
                                  CustomPriorityQueue,
                                  BestObjectiveFirstPriorityQueue,
                                  BreadthFirstPriorityQueue,
                                  DepthFirstPriorityQueue,
                                  FIFOQueue,
                                  LIFOQueue,
                                  RandomPriorityQueue,
                                  LocalGapPriorityQueue,
                                  LexicographicPriorityQueue)

class TestDispatcherQueueData(object):

    def test_bound(self):

        for sense in (minimize, maximize):
            queue = DispatcherQueueData([],
                                        None,
                                        sense)
            assert queue.nodes == []
            assert queue.worst_terminal_bound is None
            assert queue.sense == sense
            assert queue.bound() is None
            queue = DispatcherQueueData([],
                                        0,
                                        sense)
            assert queue.nodes == []
            assert queue.worst_terminal_bound == 0
            assert queue.sense == sense
            assert queue.bound() == 0
            queue = DispatcherQueueData([Node()],
                                        0,
                                        sense)
            queue.nodes[0].bound = (-1 if (sense == minimize) else 1)
            assert len(queue.nodes) == 1
            assert queue.worst_terminal_bound == 0
            assert queue.sense == sense
            assert queue.bound() == (-1 if (sense == minimize) else 1)
            queue.worst_terminal_bound = \
                (-2 if (sense == minimize) else 2)
            assert queue.bound() == (-2 if (sense == minimize) else 2)

class TestDispatcherSimple(object):

    def test_queue_strategy(self):
        node_limit = None
        time_limit = None
        queue_limit = None
        track_bound = True
        log = get_simple_logger()
        log_interval_seconds = inf
        log_new_incumbent = True
        convergence_checker = ConvergenceChecker(minimize)

        root = Node()
        root.tree_depth = 0
        root.bound = convergence_checker.unbounded_objective
        root.objective = convergence_checker.infeasible_objective
        queue = DispatcherQueueData(
            [root],
            None,
            minimize)

        disp = DispatcherLocal()
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is WorstBoundFirstPriorityQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'custom',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is CustomPriorityQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'objective',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is BestObjectiveFirstPriorityQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'breadth',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is BreadthFirstPriorityQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'depth',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is DepthFirstPriorityQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'fifo',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is FIFOQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'lifo',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is LIFOQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'random',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is RandomPriorityQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            'local_gap',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is LocalGapPriorityQueue
        disp.initialize(
            convergence_checker.infeasible_objective,
            None,
            queue,
            ('bound','local_gap'),
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is LexicographicPriorityQueue

    def test_initialize_queue(self):
        node_limit = None
        time_limit = None
        queue_limit = None
        track_bound = True
        log = get_simple_logger()
        log_interval_seconds = inf
        log_new_incumbent = True
        convergence_checker = ConvergenceChecker(minimize)

        root = Node()
        root.tree_depth = 0
        root.bound = convergence_checker.unbounded_objective
        root.objective = convergence_checker.infeasible_objective
        queue = DispatcherQueueData(
            [root],
            None,
            minimize)

        best_node_ = Node()
        best_node_.objective = 0
        best_node_._generate_uuid()
        disp = DispatcherLocal()
        disp.initialize(
            convergence_checker.infeasible_objective,
            best_node_,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == 0
        assert disp.best_node.objective == 0
        assert disp.best_node is best_node_
        disp.initialize(
            -1,
            best_node_,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == -1
        assert disp.best_node.objective == 0
        assert disp.best_node is best_node_
        best_node_.objective = 1
        disp.initialize(
            2,
            best_node_,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == 1
        assert disp.best_node.objective == 1
        assert disp.best_node is best_node_
        best_node_.objective = 1
        root.objective = -1
        disp.initialize(
            2,
            best_node_,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == 1
        assert disp.best_node.objective == 1
        assert disp.best_node is best_node_
        best_node_.objective = 1
        root.objective = -1
        disp.initialize(
            -2,
            best_node_,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            queue_limit,
            track_bound,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == -2
        assert disp.best_node.objective == 1
        assert disp.best_node is best_node_
        # bad objective sense
        queue = DispatcherQueueData(
            [root],
            None,
            maximize)
        with pytest.raises(ValueError):
            disp.initialize(
                convergence_checker.infeasible_objective,
                best_node_,
                queue,
                'bound',
                convergence_checker,
                node_limit,
                time_limit,
                queue_limit,
                track_bound,
                log,
                log_interval_seconds,
                log_new_incumbent)
