import pytest

from pybnb.common import (minimize,
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
                                  RandomPriorityQueue,
                                  LocalGapPriorityQueue)

class TestDispatcherSimple(object):

    def test_queue_strategy(self):
        node_limit = None
        time_limit = None
        log = get_simple_logger()
        log_interval_seconds = inf
        log_new_incumbent = True
        convergence_checker = ConvergenceChecker(minimize)

        root = Node(size=0)
        Node._insert_tree_id(root._data, 0)
        root.bound = convergence_checker.unbounded_objective
        root.objective = convergence_checker.infeasible_objective
        queue = DispatcherQueueData(
            nodes=[root],
            next_tree_id=1)

        disp = DispatcherLocal()
        disp.initialize(
            inf,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is WorstBoundFirstPriorityQueue
        disp.initialize(
            inf,
            queue,
            'custom',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is CustomPriorityQueue
        disp.initialize(
            inf,
            queue,
            'objective',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is BestObjectiveFirstPriorityQueue
        disp.initialize(
            inf,
            queue,
            'breadth',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is BreadthFirstPriorityQueue
        disp.initialize(
            inf,
            queue,
            'depth',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is DepthFirstPriorityQueue
        disp.initialize(
            inf,
            queue,
            'fifo',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is FIFOQueue
        disp.initialize(
            inf,
            queue,
            'random',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is RandomPriorityQueue
        disp.initialize(
            inf,
            queue,
            'local_gap',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert type(disp.queue) is LocalGapPriorityQueue

    def test_initialize_queue(self):
        node_limit = None
        time_limit = None
        log = get_simple_logger()
        log_interval_seconds = inf
        log_new_incumbent = True
        convergence_checker = ConvergenceChecker(minimize)

        root = Node(size=0)
        Node._insert_tree_id(root._data, 0)
        root.bound = convergence_checker.unbounded_objective
        root.objective = convergence_checker.infeasible_objective
        queue = DispatcherQueueData(
            nodes=[root],
            next_tree_id=1)

        disp = DispatcherLocal()
        disp.initialize(
            0,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == 0
        disp.initialize(
            1,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == 1
        root.objective = -1
        disp.initialize(
            1,
            queue,
            'bound',
            convergence_checker,
            node_limit,
            time_limit,
            log,
            log_interval_seconds,
            log_new_incumbent)
        assert disp.best_objective == -1
