import random

import pytest

from pybnb.common import (minimize,
                          maximize,
                          inf)
from pybnb.node import Node
from pybnb.priority_queue import \
    (_NoThreadingMaxPriorityFirstQueue,
     _NoThreadingFIFOQueue,
     _NoThreadingLIFOQueue,
     WorstBoundFirstPriorityQueue,
     CustomPriorityQueue,
     BestObjectiveFirstPriorityQueue,
     BreadthFirstPriorityQueue,
     DepthFirstPriorityQueue,
     FIFOQueue,
     LIFOQueue,
     RandomPriorityQueue,
     LocalGapPriorityQueue,
     LexicographicPriorityQueue,
     PriorityQueueFactory,
     register_queue_type)

def _new_child(node):
    child = Node()
    child.objective = node.objective
    child.bound = node.bound
    child.tree_depth = node.tree_depth + 1
    assert child.queue_priority is None
    assert child.state is None
    return child

def assert_isheap(x):
    for k in range(len(x)):
        if ((2*k) + 1) < len(x):
            assert x[k] <= x[2*k+1]
        if ((2*k) + 2) < len(x):
            assert x[k] <= x[2*k+2]

class TestFactory(object):

    def test_factory(self):
        assert type(PriorityQueueFactory('bound',
                                         minimize,
                                         True)) is \
            WorstBoundFirstPriorityQueue
        assert type(PriorityQueueFactory('custom',
                                         minimize,
                                         True)) is \
            CustomPriorityQueue
        assert type(PriorityQueueFactory('objective',
                                         minimize,
                                         True)) is \
            BestObjectiveFirstPriorityQueue
        assert type(PriorityQueueFactory('breadth',
                                         minimize,
                                         True)) is \
            BreadthFirstPriorityQueue
        assert type(PriorityQueueFactory('depth',
                                         minimize,
                                         True)) is \
            DepthFirstPriorityQueue
        assert type(PriorityQueueFactory('fifo',
                                         minimize,
                                         True)) is \
            FIFOQueue
        assert type(PriorityQueueFactory('lifo',
                                         minimize,
                                         True)) is \
            LIFOQueue
        assert type(PriorityQueueFactory('random',
                                         minimize,
                                         True)) is \
            RandomPriorityQueue
        assert type(PriorityQueueFactory('local_gap',
                                         minimize,
                                         True)) is \
            LocalGapPriorityQueue
        with pytest.raises(ValueError):
            PriorityQueueFactory('_not_a_type_',
                                 minimize,
                                 True)
        # test LexicographicPriorityQueue creation
        assert type(PriorityQueueFactory(('bound','objective'),
                                         minimize,
                                         True)) is \
            LexicographicPriorityQueue
        with pytest.raises(ValueError):
            PriorityQueueFactory(('_not_a_type_',),
                                 minimize,
                                 True)
        with pytest.raises(ValueError):
            PriorityQueueFactory(('custom',),
                                 minimize,
                                 True)
        with pytest.raises(ValueError):
            PriorityQueueFactory(('fifo','custom'),
                                 minimize,
                                 True)
        with pytest.raises(ValueError):
            PriorityQueueFactory(('custom','fifo'),
                                 minimize,
                                 True)
        with pytest.raises(ValueError):
            PriorityQueueFactory((),
                                 minimize,
                                 True)

    def test_register_queue_type(self):
        assert PriorityQueueFactory._types['bound'] is \
            WorstBoundFirstPriorityQueue
        # its okay to re-register the exact same thing
        register_queue_type('bound',
                            WorstBoundFirstPriorityQueue)
        assert PriorityQueueFactory._types['bound'] is \
            WorstBoundFirstPriorityQueue
        with pytest.raises(ValueError):
            register_queue_type('bound', None)
        assert PriorityQueueFactory._types['bound'] is \
            WorstBoundFirstPriorityQueue
        assert '_not_a_type_' not in PriorityQueueFactory._types
        try:
            register_queue_type('_not_a_type_', None)
            assert PriorityQueueFactory._types['_not_a_type_'] is None
        finally:
            PriorityQueueFactory._types.pop('_not_a_type_',None)

class Test_NoThreadingMaxPriorityFirstQueue(object):

    def test_size(self):
        q = _NoThreadingMaxPriorityFirstQueue()
        assert_isheap(q._heap)
        assert q.size() == 0
        with pytest.raises(IndexError):
            q.next()
        cntr = {}
        for i in range(10):
            cntr[i] = q.put(i,0)
            assert q.size() == i+1
            assert_isheap(q._heap)
        for i in range(10):
            c_, x_ = q.next()
            assert c_ == cntr[x_]
            x = q.get()
            assert x_ == x
            assert x == i
            assert q.size() == 10-(i+1)
            assert_isheap(q._heap)
        assert q.size() == 0
        assert q.get() is None
        with pytest.raises(IndexError):
            q.next()
        assert_isheap(q._heap)

    def test_put_get(self):
        q = _NoThreadingMaxPriorityFirstQueue()
        with pytest.raises(ValueError):
            q.put(None, 0)
        with pytest.raises(ValueError):
            q.put_get(None, 0)
        assert sorted(q.items()) == []
        assert_isheap(q._heap)
        assert q.put_get(1,1) == (0,1)
        assert q.put_get(1,1) == (1,1)
        cntr = {}
        cntr[1] = q.put(1,1)
        assert sorted(q.items()) == [1]
        assert_isheap(q._heap)
        cntr[2] = q.put(2,2)
        assert sorted(q.items()) == [1,2]
        assert_isheap(q._heap)
        cntr[3] = q.put(3,2)
        assert sorted(q.items()) == [1,2,3]
        assert_isheap(q._heap)
        cntr[4], item_ = q.put_get(4,-4)
        assert item_ == 2
        assert sorted(q.items()) == [1,3,4]
        assert_isheap(q._heap)
        assert q.next() == (cntr[q.next()[1]], 3)
        assert q.get() == 3
        assert sorted(q.items()) == [1,4]
        assert_isheap(q._heap)
        assert q.next() == (cntr[q.next()[1]], 1)
        assert q.get() == 1
        assert sorted(q.items()) == [4]
        assert_isheap(q._heap)
        assert q.next() == (cntr[q.next()[1]], 4)
        assert q.get() == 4
        assert sorted(q.items()) == []
        assert_isheap(q._heap)
        with pytest.raises(IndexError):
            q.next()

    def test_filter(self):
        for k in range(-10,11):
            cutoff = k*11
            def _filter(item):
                if item <= cutoff:
                    return True
            items = list(range(-1000,1000))
            random.shuffle(items)
            q = _NoThreadingMaxPriorityFirstQueue()
            for i in items:
                q.put(i, 1)
            assert_isheap(q._heap)
            correct = []
            removed = []
            for _,_,item in q._heap:
                if item <= cutoff:
                    correct.append(item)
                else:
                    removed.append(item)
            removed_ = q.filter(_filter)
            assert removed_ == removed
            assert_isheap(q._heap)
            check = []
            for _,_,item in q._heap:
                check.append(item)
            assert sorted(correct) == sorted(check)

class Test_NoThreadingFIFOQueue(object):

    def test_size(self):
        q = _NoThreadingFIFOQueue()
        assert q.size() == 0
        with pytest.raises(IndexError):
            q.next()
        cntr = {}
        for i in range(10):
            cntr[i] = q.put(i)
            assert q.size() == i+1
        for i in range(10):
            c_, x_ = q.next()
            assert c_ == cntr[x_]
            x = q.get()
            assert x_ == x
            assert x == i
            assert q.size() == 10-(i+1)
        assert q.size() == 0
        assert q.get() is None
        with pytest.raises(IndexError):
            q.next()

    def test_put_get(self):
        q = _NoThreadingFIFOQueue()
        with pytest.raises(ValueError):
            q.put(None)
        with pytest.raises(ValueError):
            q.put_get(None)
        assert list(q.items()) == []
        assert q.put_get(1) == (0,1)
        assert q.put_get(1) == (1,1)
        cntr = {}
        cntr[1] = q.put(1)
        assert list(q.items()) == [1]
        cntr[2] = q.put(2)
        assert list(q.items()) == [1,2]
        cntr[3] = q.put(3)
        assert list(q.items()) == [1,2,3]
        cntr[4], item_ = q.put_get(4)
        assert item_ == 1
        assert list(q.items()) == [2,3,4]
        assert q.next() == (cntr[q.next()[1]], 2)
        assert q.get() == 2
        assert list(q.items()) == [3,4]
        assert q.next() == (cntr[q.next()[1]], 3)
        assert q.get() == 3
        assert list(q.items()) == [4]
        assert q.next() == (cntr[q.next()[1]], 4)
        assert q.get() == 4
        assert list(q.items()) == []
        with pytest.raises(IndexError):
            q.next()

    def test_filter(self):
        for k in range(-10,11):
            cutoff = k*11
            def _filter(item):
                if item <= cutoff:
                    return True
            items = list(range(-1000,1000))
            random.shuffle(items)
            q = _NoThreadingFIFOQueue()
            for i in items:
                q.put(i)
            correct = []
            removed = []
            for _,item in q._deque:
                if item <= cutoff:
                    correct.append(item)
                else:
                    removed.append(item)
            removed_ = q.filter(_filter)
            assert removed_ == removed
            check = []
            for _,item in q._deque:
                check.append(item)
            assert sorted(correct) == sorted(check)
        for k in range(-10,11):
            cutoff = k*11
            def _filter(item):
                if item <= cutoff:
                    return True
            items = list(range(-1000,1000))
            random.shuffle(items)
            q = _NoThreadingFIFOQueue()
            for i in items:
                q.put(i)
            correct = []
            removed = []
            for cnt,item in q._deque:
                if item <= cutoff:
                    correct.append(item)
                else:
                    removed.append((cnt,item))
            removed_ = q.filter(_filter, include_counters=True)
            assert removed_ == removed
            check = []
            for _,item in q._deque:
                check.append(item)
            assert sorted(correct) == sorted(check)

class Test_NoThreadingLIFOQueue(object):

    def test_size(self):
        q = _NoThreadingLIFOQueue()
        assert q.size() == 0
        with pytest.raises(IndexError):
            q.next()
        cntr = {}
        for i in range(10):
            cntr[i] = q.put(i)
            assert q.size() == i+1
        for i in reversed(range(10)):
            c_, x_ = q.next()
            assert c_ == cntr[x_]
            x = q.get()
            assert x_ == x
            assert x == i
            assert q.size() == i
        assert q.size() == 0
        assert q.get() is None
        with pytest.raises(IndexError):
            q.next()

    def test_put_get(self):
        q = _NoThreadingLIFOQueue()
        with pytest.raises(ValueError):
            q.put(None)
        with pytest.raises(ValueError):
            q.put_get(None)
        assert list(q.items()) == []
        assert q.put_get(1) == (0,1)
        assert q.put_get(1) == (1,1)
        cntr = {}
        cntr[1] = q.put(1)
        assert list(q.items()) == [1]
        cntr[2] = q.put(2)
        assert list(q.items()) == [1,2]
        cntr[3] = q.put(3)
        assert list(q.items()) == [1,2,3]
        cntr[4], item_ = q.put_get(4)
        assert item_ == 4
        assert list(q.items()) == [1,2,3]
        assert q.next() == (cntr[q.next()[1]], 3)
        assert q.get() == 3
        assert list(q.items()) == [1,2]
        assert q.next() == (cntr[q.next()[1]], 2)
        assert q.get() == 2
        assert list(q.items()) == [1]
        assert q.next() == (cntr[q.next()[1]], 1)
        assert q.get() == 1
        assert list(q.items()) == []
        with pytest.raises(IndexError):
            q.next()

    def test_filter(self):
        for k in range(-10,11):
            cutoff = k*11
            def _filter(item):
                if item <= cutoff:
                    return True
            items = list(range(-1000,1000))
            random.shuffle(items)
            q = _NoThreadingLIFOQueue()
            for i in items:
                q.put(i)
            correct = []
            removed = []
            for _,item in q._items:
                if item <= cutoff:
                    correct.append(item)
                else:
                    removed.append(item)
            removed_ = q.filter(_filter)
            assert removed_ == removed
            check = []
            for _,item in q._items:
                check.append(item)
            assert sorted(correct) == sorted(check)
        for k in range(-10,11):
            cutoff = k*11
            def _filter(item):
                if item <= cutoff:
                    return True
            items = list(range(-1000,1000))
            random.shuffle(items)
            q = _NoThreadingLIFOQueue()
            for i in items:
                q.put(i)
            correct = []
            removed = []
            for cnt,item in q._items:
                if item <= cutoff:
                    correct.append(item)
                else:
                    removed.append((cnt,item))
            removed_ = q.filter(_filter, include_counters=True)
            assert removed_ == removed
            check = []
            for _,item in q._items:
                check.append(item)
            assert sorted(correct) == sorted(check)

def _check_items(q, items):
    found = dict((id(item), False)
                 for item in items)
    assert len(list(q.items())) == len(items)
    for queue_item in q.items():
        assert id(queue_item) in found
        assert not found[id(queue_item)]
        found[id(queue_item)] = True
    assert all(found.values())

class TestWorstBoundFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = WorstBoundFirstPriorityQueue(sense=minimize,
                                         track_bound=True)
        node = Node()
        node.bound = 1
        assert node.queue_priority is None
        q.put(node)
        assert node.queue_priority == -1

        q = WorstBoundFirstPriorityQueue(sense=maximize,
                                         track_bound=True)
        node = Node()
        node.bound = 1
        assert node.queue_priority is None
        q.put(node)
        assert node.queue_priority == 1

    def test_usage_minimize(self):
        q = WorstBoundFirstPriorityQueue(sense=minimize,
                                         track_bound=True)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node()
            node.bound = i
            assert node.queue_priority is None
            cnt_ = q.put(node)
            assert cnt_ == i-1
            assert node.queue_priority == -node.bound
            items.append(node)
            assert q.size() == i
            assert q.bound() == 1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == 1
        removed = q.filter(lambda n_: n_.bound >= 5)
        assert q.size() == 6
        assert len(removed) == 4
        for node_ in removed:
            assert node_.bound < 5
        assert q.bound() == 5
        for i in range(5, 11):
            node = q.get()
            assert node.bound == i
            assert node.queue_priority == -node.bound
            if i != 10:
                assert q.bound() == i+1
            else:
                assert q.bound() is None
        assert q.size() == 0
        node = Node()
        node.bound = -1
        assert node.queue_priority is None
        cnt_ = q.put(node)
        node_ = q.get()
        assert cnt_ == 10
        assert node_ is node
        assert node.queue_priority == 1
        node.bound = -2
        cnt_ = q.put(node)
        assert node.queue_priority == 2
        assert cnt_ == 11
        node2 = Node()
        node2.bound = -3
        assert node2.queue_priority is None
        cnt_ = q.put(node2)
        node_ = q.get()
        assert cnt_ == 12
        assert node_ is node2
        assert node2.queue_priority == 3
        node2.bound = -1
        cnt_ = q.put(node2)
        node_ = q.get()
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert node_ is node
        assert q.size() == 1

    def test_usage_maximize(self):
        q = WorstBoundFirstPriorityQueue(sense=maximize,
                                         track_bound=True)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node()
            node.bound = -i
            assert node.queue_priority is None
            cnt_ = q.put(node)
            assert cnt_ == i-1
            assert node.queue_priority == node.bound
            items.append(node)
            assert q.size() == i
            assert q.bound() == -1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == -1
        removed = q.filter(lambda n_: n_.bound <= -5)
        assert q.size() == 6
        assert len(removed) == 4
        for node_ in removed:
            assert node_.bound > -5
        assert q.bound() == -5
        for i in range(5, 11):
            node = q.get()
            assert node.bound == -i
            assert node.queue_priority == node.bound
            if i != 10:
                assert q.bound() == -i-1
            else:
                assert q.bound() is None
        assert q.size() == 0
        node = Node()
        node.bound = 1
        assert node.queue_priority is None
        cnt_ = q.put(node)
        node_ = q.get()
        assert cnt_ == 10
        assert node_ is node
        assert node.queue_priority == 1
        node.bound = 2
        cnt_ = q.put(node)
        assert node.queue_priority == 2
        assert cnt_ == 11
        node2 = Node()
        node2.bound = 3
        assert node2.queue_priority is None
        cnt_ = q.put(node2)
        node_ = q.get()
        assert cnt_ == 12
        assert node_ is node2
        assert node2.queue_priority == 3
        node2.bound = 1
        cnt_ = q.put(node2)
        node_ = q.get()
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert node_ is node
        assert q.size() == 1

class TestCustomPriorityQueue(object):

    def test_missing_queue_priority(self):
        q = CustomPriorityQueue(sense=minimize,
                                track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        assert node.queue_priority is None
        with pytest.raises(ValueError):
            q.put(node)
        node.queue_priority = 1
        q.put(node)
        child = _new_child(node)
        assert child.queue_priority is None
        with pytest.raises(ValueError):
            q.put(child)

    def test_usage_minimize(self):
        q = CustomPriorityQueue(sense=minimize,
                                track_bound=True)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node()
            node.bound = i
            node.queue_priority = -i
            assert q.put(node) == i-1
            assert node.queue_priority == -i
            items.append(node)
            assert q.size() == i
            assert q.bound() == 1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == 1
        removed = q.filter(lambda n_: n_.bound >= 5)
        assert q.size() == 6
        assert len(removed) == 4
        for node_ in removed:
            assert node_.bound < 5
        assert q.bound() == 5
        for i in range(5, 11):
            node = q.get()
            assert node.bound == i
            assert node.queue_priority == -i
            if i != 10:
                assert q.bound() == i+1
            else:
                assert q.bound() is None
        assert q.size() == 0
        node = Node()
        node.bound = 0
        node.queue_priority = 1
        cnt_ = q.put(node)
        node_ = q.get()
        assert cnt_ == 10
        assert node_ is node
        assert node.queue_priority == 1
        assert q.bound() is None
        node.queue_priority = 2
        cnt_ = q.put(node)
        assert node.queue_priority == 2
        assert cnt_ == 11
        assert q.bound() == 0
        node2 = Node()
        node2.bound = 1
        node2.queue_priority = 3
        cnt_ = q.put(node2)
        node_ = q.get()
        assert cnt_ == 12
        assert node_ is node2
        assert node2.queue_priority == 3
        assert q.bound() == 0
        node2.queue_priority = 1
        cnt_ = q.put(node2)
        node_ = q.get()
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert node_ is node
        assert q.size() == 1
        assert q.bound() == 1
        # no bound tracking
        q = CustomPriorityQueue(sense=minimize,
                                track_bound=False)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node()
            node.bound = i
            node.queue_priority = -i
            assert q.put(node) == i-1
            assert node.queue_priority == -i
            items.append(node)
            assert q.size() == i
            assert q.bound() == -inf
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == -inf
        removed = q.filter(lambda n_: n_.bound >= 5)
        assert q.size() == 6
        assert len(removed) == 4
        for node_ in removed:
            assert node_.bound < 5
        assert q.bound() == -inf
        for i in range(5, 11):
            node = q.get()
            assert node.bound == i
            assert node.queue_priority == -i
            if i != 10:
                assert q.bound() == -inf
            else:
                assert q.bound() is None
        assert q.size() == 0

    def test_usage_maximize(self):
        q = CustomPriorityQueue(sense=maximize,
                                track_bound=True)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node()
            node.bound = -i
            node.queue_priority = i
            assert q.put(node) == i-1
            assert node.queue_priority == i
            items.append(node)
            assert q.size() == i
            assert q.bound() == -1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == -1
        removed = q.filter(lambda n_: n_.bound <= -5)
        assert q.size() == 6
        assert len(removed) == 4
        for node_ in removed:
            assert node_.bound > -5
        assert q.bound() == -5
        for i in range(10, 4, -1):
            node = q.get()
            assert node.bound == -i
            assert node.queue_priority == i
            if i != 5:
                assert q.bound() == -5
            else:
                assert q.bound() is None
        assert q.size() == 0
        node = Node()
        node.bound = 0
        node.queue_priority = 1
        cnt_ = q.put(node)
        node_ = q.get()
        assert cnt_ == 10
        assert node_ is node
        assert node.queue_priority == 1
        assert q.bound() is None
        node.queue_priority = 2
        cnt_ = q.put(node)
        assert node.queue_priority == 2
        assert cnt_ == 11
        assert q.bound() == 0
        node2 = Node()
        node2.bound = 1
        node2.queue_priority = 3
        cnt_ = q.put(node2)
        node_ = q.get()
        assert cnt_ == 12
        assert node_ is node2
        assert node2.queue_priority == 3
        assert q.bound() == 0
        node2.queue_priority = 1
        cnt_ = q.put(node2)
        node_ = q.get()
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert node_ is node
        assert q.size() == 1
        assert q.bound() == 1
        # no bound tracking
        q = CustomPriorityQueue(sense=maximize,
                                track_bound=False)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node()
            node.bound = -i
            node.queue_priority = i
            assert q.put(node) == i-1
            assert node.queue_priority == i
            items.append(node)
            assert q.size() == i
            assert q.bound() == inf
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == inf
        removed = q.filter(lambda n_: n_.bound <= -5)
        assert q.size() == 6
        assert len(removed) == 4
        for node_ in removed:
            assert node_.bound > -5
        assert q.bound() == inf
        for i in range(10, 4, -1):
            node = q.get()
            assert node.bound == -i
            assert node.queue_priority == i
            if i != 5:
                assert q.bound() == inf
            else:
                assert q.bound() is None
        assert q.size() == 0

class TestBestObjectiveFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = BestObjectiveFirstPriorityQueue(sense=minimize,
                                            track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = -1
        assert node.queue_priority is None
        node.objective = 1
        assert q.put(node) == 0
        assert node.objective == 1
        assert node.queue_priority == -1
        child = _new_child(node)
        assert child.objective == 1
        child.objective = 0
        assert child.queue_priority is None
        cnt = q.put(child)
        node_ = q.get()
        assert child.queue_priority == 0
        assert cnt == 1
        assert node_ is child
        child.objective = 2
        cnt = q.put(child)
        node_ = q.get()
        assert child.queue_priority == -2
        assert cnt == 2
        assert node_ is node
        assert q.bound() == -1

        q = BestObjectiveFirstPriorityQueue(sense=maximize,
                                            track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = 3
        assert node.queue_priority is None
        node.objective = 1
        assert q.put(node) == 0
        assert node.objective == 1
        assert node.queue_priority == 1
        child = _new_child(node)
        assert child.objective == 1
        child.objective = 2
        assert child.queue_priority is None
        cnt = q.put(child)
        node_ = q.get()
        assert child.queue_priority == 2
        assert cnt == 1
        assert node_ is child
        child.objective = 0
        cnt = q.put(child)
        node_ = q.get()
        assert child.queue_priority == 0
        assert cnt == 2
        assert node_ is node
        assert q.bound() == 3

class TestBreadthFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = BreadthFirstPriorityQueue(sense=minimize,
                                      track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.tree_depth == 0
        assert node.queue_priority == 0
        child = _new_child(node)
        assert child.tree_depth == 1
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority == -child.tree_depth

        l1 = Node()
        l1.tree_depth = 0
        l1.bound = 1
        l2 = _new_child(l1)
        l3 = _new_child(l2)
        q = BreadthFirstPriorityQueue(sense=minimize,
                                      track_bound=True)
        q.put(l2)
        cnt = q.put(l1)
        node_ = q.get()
        assert cnt == 1
        assert node_ is l1
        cnt = q.put(l3)
        node_ = q.get()
        assert cnt == 2
        assert node_ is l2
        assert q.bound() == 1

class TestDepthFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = DepthFirstPriorityQueue(sense=minimize,
                                    track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.tree_depth == 0
        assert node.queue_priority == 0
        child = _new_child(node)
        assert child.tree_depth == 1
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority == child.tree_depth

        l1 = Node()
        l1.tree_depth = 0
        l1.bound = 1
        l2 = _new_child(l1)
        l3 = _new_child(l2)
        q = DepthFirstPriorityQueue(sense=minimize,
                                    track_bound=True)
        q.put(l2)
        cnt = q.put(l3)
        node_ = q.get()
        assert cnt == 1
        assert node_ is l3
        cnt = q.put(l2)
        node_ = q.get()
        assert cnt == 2
        assert node_ is l2
        assert q.bound() == 1

class TestFIFOQueue(object):

    def test_overwrites_queue_priority(self):
        q = FIFOQueue(sense=minimize,
                      track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.queue_priority == 0
        child = _new_child(node)
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority == -1

        l1 = Node()
        l1.tree_depth = 0
        l1.bound = 1
        l2 = _new_child(l1)
        l3 = _new_child(l2)
        q = FIFOQueue(sense=minimize,
                      track_bound=True)
        cnt = q.put(l2)
        node_ = q.get()
        assert cnt == 0
        assert node_ is l2
        cnt = q.put(l2)
        assert cnt == 1
        cnt = q.put(l3)
        node_ = q.get()
        assert cnt == 2
        assert node_ is l2
        cnt = q.put(l1)
        node_ = q.get()
        assert cnt == 3
        assert node_ is l3
        assert q.bound() == 1

class TestLIFOQueue(object):

    def test_overwrites_queue_priority(self):
        q = LIFOQueue(sense=minimize,
                      track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.queue_priority == 0
        child = _new_child(node)
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority == 1

        l1 = Node()
        l1.tree_depth = 0
        l1.bound = 1
        l2 = _new_child(l1)
        l3 = _new_child(l2)
        q = LIFOQueue(sense=minimize,
                      track_bound=True)
        cnt = q.put(l2)
        node_ = q.get()
        assert cnt == 0
        assert node_ is l2
        cnt = q.put(l2)
        assert q.bound() == 1
        assert cnt == 1
        cnt = q.put(l3)
        node_ = q.get()
        assert cnt == 2
        assert node_ is l3
        node_ = q.get()
        assert node_ is l2
        assert q.bound() is None

class TestRandomPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = RandomPriorityQueue(sense=minimize,
                                track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.queue_priority is not None
        assert 0 <= node.queue_priority <= 1
        child = _new_child(node)
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority is not None
        assert 0 <= child.queue_priority <= 1

        l1 = Node()
        l1.tree_depth = 0
        l1.bound = 1
        l2 = _new_child(l1)
        l3 = _new_child(l2)
        q = RandomPriorityQueue(sense=minimize,
                                track_bound=True)
        assert l2.queue_priority is None
        cnt = q.put(l2)
        node_ = q.get()
        assert node_ is l2
        assert l2.queue_priority is not None
        assert 0 <= l2.queue_priority <= 1
        assert cnt == 0
        cnt = q.put(l2)
        assert cnt == 1
        assert l3.queue_priority is None
        cnt = q.put(l3)
        node_ = q.get()
        assert cnt == 2
        assert l3.queue_priority is not None
        assert 0 <= l3.queue_priority <= 1
        assert node_ is max([l2, l3],
                            key=lambda x_: x_.queue_priority)

class TestLocalGapPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = LocalGapPriorityQueue(sense=minimize,
                                  track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = -inf
        node.objective = inf
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.queue_priority is not None
        assert node.queue_priority == inf
        child = _new_child(node)
        assert child.bound == -inf
        assert child.objective == inf
        child.bound = 0
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority is not None
        assert child.queue_priority == inf
        child = _new_child(child)
        assert child.bound == 0
        assert child.objective == inf
        child.objective = 1
        assert child.queue_priority is None
        assert q.put(child) == 2
        assert child.queue_priority is not None
        assert child.queue_priority == 1

        l1 = Node()
        l1.tree_depth = 0
        l1.bound = 1
        l1.objective = 5
        l2 = _new_child(l1)
        l3 = _new_child(l2)
        q = LocalGapPriorityQueue(sense=minimize,
                                  track_bound=True)
        assert l2.queue_priority is None
        cnt = q.put(l2)
        node_ = q.get()
        assert node_ is l2
        assert l2.queue_priority is not None
        assert l2.queue_priority == 4
        assert cnt == 0
        cnt = q.put(l2)
        assert cnt == 1
        assert l3.queue_priority is None
        l3.objective = 6
        cnt = q.put(l3)
        node_ = q.get()
        assert cnt == 2
        assert l3.queue_priority is not None
        assert l3.queue_priority == 5
        assert node_ is l3

        q = LocalGapPriorityQueue(sense=maximize,
                                  track_bound=True)
        node = Node()
        node.tree_depth = 0
        node.bound = inf
        node.objective = -inf
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.queue_priority is not None
        assert node.queue_priority == inf
        child = _new_child(node)
        assert child.bound == inf
        assert child.objective == -inf
        child.bound = 0
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority is not None
        assert child.queue_priority == inf
        child = _new_child(child)
        assert child.bound == 0
        assert child.objective == -inf
        child.objective = -1
        assert child.queue_priority is None
        assert q.put(child) == 2
        assert child.queue_priority is not None
        assert child.queue_priority == 1

        l1 = Node()
        l1.tree_depth = 0
        l1.bound = -1
        l1.objective = -5
        l2 = _new_child(l1)
        l3 = _new_child(l2)
        q = LocalGapPriorityQueue(sense=maximize,
                                  track_bound=True)
        assert l2.queue_priority is None
        cnt = q.put(l2)
        node_ = q.get()
        assert node_ is l2
        assert l2.queue_priority is not None
        assert l2.queue_priority == 4
        assert cnt == 0
        cnt = q.put(l2)
        assert cnt == 1
        assert l3.queue_priority is None
        l3.objective = -6
        cnt = q.put(l3)
        node_ = q.get()
        assert cnt == 2
        assert l3.queue_priority is not None
        assert l3.queue_priority == 5
        assert node_ is l3

class TestLexicographicPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        # min
        q = LexicographicPriorityQueue(
            (WorstBoundFirstPriorityQueue,
             BestObjectiveFirstPriorityQueue),
            minimize,
            True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        node.objective = 2
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.queue_priority is not None
        assert node.queue_priority == (0,-2)
        c1 = child = _new_child(node)
        assert child.bound == 0
        assert child.objective == 2
        child.objective = 1
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority is not None
        assert child.queue_priority == (0,-1)
        c2 = child = _new_child(child)
        assert child.bound == 0
        assert child.objective == 1
        child.bound = -1
        child.objective = 2
        assert child.queue_priority is None
        assert q.put(child) == 2
        assert child.queue_priority is not None
        assert child.queue_priority == (1,-2)
        c3 = child = _new_child(child)
        assert child.bound == -1
        assert child.objective == 2
        child.bound = 1
        child.objective = -100
        assert child.queue_priority is None
        assert q.put(child) == 3
        assert child.queue_priority is not None
        assert child.queue_priority == (-1,100)
        assert q.get() is c2
        assert q.get() is c1
        assert q.get() is node
        assert q.get() is c3
        assert q.get() is None

        # max
        q = LexicographicPriorityQueue(
            (WorstBoundFirstPriorityQueue,
             BestObjectiveFirstPriorityQueue),
            maximize,
            True)
        node = Node()
        node.tree_depth = 0
        node.bound = 0
        node.objective = -2
        assert node.queue_priority is None
        assert q.put(node) == 0
        assert node.queue_priority is not None
        assert node.queue_priority == (0,-2)
        c1 = child = _new_child(node)
        assert child.bound == 0
        assert child.objective == -2
        child.objective = -1
        assert child.queue_priority is None
        assert q.put(child) == 1
        assert child.queue_priority is not None
        assert child.queue_priority == (0,-1)
        c2 = child = _new_child(child)
        assert child.bound == 0
        assert child.objective == -1
        child.bound = 1
        child.objective = -2
        assert child.queue_priority is None
        assert q.put(child) == 2
        assert child.queue_priority is not None
        assert child.queue_priority == (1,-2)
        c3 = child = _new_child(child)
        assert child.bound == 1
        assert child.objective == -2
        child.bound = -1
        child.objective = 100
        assert child.queue_priority is None
        assert q.put(child) == 3
        assert child.queue_priority is not None
        assert child.queue_priority == (-1,100)
        assert q.get() is c2
        assert q.get() is c1
        assert q.get() is node
        assert q.get() is c3
        assert q.get() is None
