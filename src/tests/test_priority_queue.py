import random

import pytest

from pybnb.common import (minimize,
                          maximize)
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.node import Node
from pybnb.priority_queue import \
    (_NoThreadingMaxPriorityFirstQueue,
     _NoThreadingFIFOQueue,
     WorstBoundFirstPriorityQueue,
     CustomPriorityQueue,
     BreadthFirstPriorityQueue,
     DepthFirstPriorityQueue,
     FIFOQueue)

def assert_isheap(x):
    for k in range(len(x)):
        if ((2*k) + 1) < len(x):
            assert x[k] <= x[2*k+1]
        if ((2*k) + 2) < len(x):
            assert x[k] <= x[2*k+2]

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
            assert x_ is x
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
        with pytest.raises(AssertionError):
            q.put(None, 0)
        assert sorted(q.items()) == []
        assert_isheap(q._heap)
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
        cntr[4] = q.put(4,-4)
        assert sorted(q.items()) == [1,2,3,4]
        assert_isheap(q._heap)
        assert q.next() == (cntr[q.next()[1]], 2)
        assert q.get() == 2
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
            assert x_ is x
            assert x == i
            assert q.size() == 10-(i+1)
        assert q.size() == 0
        assert q.get() is None
        with pytest.raises(IndexError):
            q.next()

    def test_put_get(self):
        q = _NoThreadingFIFOQueue()
        with pytest.raises(AssertionError):
            q.put(None)
        assert list(q.items()) == []
        cntr = {}
        cntr[1] = q.put(1)
        assert list(q.items()) == [1]
        cntr[2] = q.put(2)
        assert list(q.items()) == [1,2]
        cntr[3] = q.put(3)
        assert list(q.items()) == [1,2,3]
        cntr[4] = q.put(4)
        assert list(q.items()) == [1,2,3,4]
        assert q.next() == (cntr[q.next()[1]], 1)
        assert q.get() == 1
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
        q = WorstBoundFirstPriorityQueue(minimize)
        node = Node(size=0)
        node.bound = 1
        assert node.queue_priority is None
        q.put(node._data)
        assert node.queue_priority == -1

        q = WorstBoundFirstPriorityQueue(maximize)
        node = Node(size=0)
        node.bound = 1
        assert node.queue_priority is None
        q.put(node._data)
        assert node.queue_priority == 1

    def test_usage_minimize(self):
        q = WorstBoundFirstPriorityQueue(minimize)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node(size=0)
            node.bound = i
            assert node.queue_priority is None
            q.put(node._data)
            assert node.queue_priority == -node.bound
            items.append(node._data)
            assert q.size() == i
            assert q.bound() == 1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == 1
        removed = q.filter(lambda data: \
                           Node._extract_bound(data) >= 5)
        assert q.size() == 6
        assert len(removed) == 4
        for data in removed:
            assert Node._extract_bound(data) < 5
        assert q.bound() == 5
        for i in range(5, 11):
            node = Node(data_=q.get())
            assert node.bound == i
            assert node.queue_priority == -node.bound
            if i != 10:
                assert q.bound() == i+1
            else:
                assert q.bound() is None
        assert q.size() == 0

    def test_usage_maximize(self):
        q = WorstBoundFirstPriorityQueue(maximize)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node(size=0)
            node.bound = -i
            assert node.queue_priority is None
            q.put(node._data)
            assert node.queue_priority == node.bound
            items.append(node._data)
            assert q.size() == i
            assert q.bound() == -1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == -1
        removed = q.filter(lambda data: \
                           Node._extract_bound(data) <= -5)
        assert q.size() == 6
        assert len(removed) == 4
        for data in removed:
            assert Node._extract_bound(data) > -5
        assert q.bound() == -5
        for i in range(5, 11):
            node = Node(data_=q.get())
            assert node.bound == -i
            assert node.queue_priority == node.bound
            if i != 10:
                assert q.bound() == -i-1
            else:
                assert q.bound() is None
        assert q.size() == 0

class TestCustomPriorityQueue(object):

    def test_missing_queue_priority(self):
        q = CustomPriorityQueue(minimize)
        node = Node(size=0)
        assert node.queue_priority is None
        with pytest.raises(ValueError):
            q.put(node._data)
        node.queue_priority = 1
        q.put(node._data)

    def test_usage_minimize(self):
        q = CustomPriorityQueue(minimize)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node(size=0)
            node.bound = i
            node.queue_priority = -i
            assert q.put(node._data) == i-1
            assert node.queue_priority == -i
            items.append(node._data)
            assert q.size() == i
            assert q.bound() == 1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == 1
        removed = q.filter(lambda data: \
                           Node._extract_bound(data) >= 5)
        assert q.size() == 6
        assert len(removed) == 4
        for data in removed:
            assert Node._extract_bound(data) < 5
        assert q.bound() == 5
        for i in range(5, 11):
            node = Node(data_=q.get())
            assert node.bound == i
            assert node.queue_priority == -i
            if i != 10:
                assert q.bound() == i+1
            else:
                assert q.bound() is None
        assert q.size() == 0

    def test_usage_maximize(self):
        q = CustomPriorityQueue(maximize)
        assert q.size() == 0
        assert q.bound() is None
        assert len(list(q.items())) == 0
        assert q.get() is None
        items = []
        for i in range(1,11):
            node = Node(size=0)
            node.bound = -i
            node.queue_priority = i
            assert q.put(node._data) == i-1
            assert node.queue_priority == i
            items.append(node._data)
            assert q.size() == i
            assert q.bound() == -1
            _check_items(q, items)
        assert q.size() == 10
        assert q.bound() == -1
        removed = q.filter(lambda data: \
                           Node._extract_bound(data) <= -5)
        assert q.size() == 6
        assert len(removed) == 4
        for data in removed:
            assert Node._extract_bound(data) > -5
        assert q.bound() == -5
        for i in range(10, 4, -1):
            node = Node(data_=q.get())
            assert node.bound == -i
            assert node.queue_priority == i
            if i != 5:
                assert q.bound() == -5
            else:
                assert q.bound() is None
        assert q.size() == 0

class TestBreadthFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = BreadthFirstPriorityQueue(minimize)
        node = Node(size=0)
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.tree_depth == 0
        assert node.queue_priority == 0
        child = node.new_child()
        assert child.tree_depth == 1
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority == -child.tree_depth

class TestDepthFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = DepthFirstPriorityQueue(minimize)
        node = Node(size=0)
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.tree_depth == 0
        assert node.queue_priority == 0
        child = node.new_child()
        assert child.tree_depth == 1
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority == child.tree_depth

class TestFIFOQueue(object):

    def test_overwrites_queue_priority(self):
        q = FIFOQueue(minimize)
        node = Node(size=0)
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.queue_priority == 0
        child = node.new_child()
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority == -1
