import random

import pytest

from pybnb.common import (minimize,
                          maximize,
                          inf)
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.node import Node
from pybnb.priority_queue import \
    (_NoThreadingMaxPriorityFirstQueue,
     _NoThreadingFIFOQueue,
     WorstBoundFirstPriorityQueue,
     CustomPriorityQueue,
     BestObjectiveFirstPriorityQueue,
     BreadthFirstPriorityQueue,
     DepthFirstPriorityQueue,
     FIFOQueue,
     RandomPriorityQueue,
     LocalGapPriorityQueue)

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
        with pytest.raises(AssertionError):
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
        with pytest.raises(AssertionError):
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
            cnt_ = q.put(node._data)
            assert cnt_ == i-1
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
        node = Node(size=0)
        node.bound = -1
        assert node.queue_priority is None
        cnt_, data = q.put_get(node._data)
        assert cnt_ == 10
        assert data is node._data
        assert node.queue_priority == 1
        node.bound = -2
        cnt_ = q.put(node._data)
        assert node.queue_priority == 2
        assert cnt_ == 11
        node2 = Node(size=0)
        node2.bound = -3
        assert node2.queue_priority is None
        cnt_, data = q.put_get(node2._data)
        assert cnt_ == 12
        assert data is node2._data
        assert node2.queue_priority == 3
        node2.bound = -1
        cnt_, data = q.put_get(node2._data)
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert data is node._data
        assert q.size() == 1

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
            cnt_ = q.put(node._data)
            assert cnt_ == i-1
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
        node = Node(size=0)
        node.bound = 1
        assert node.queue_priority is None
        cnt_, data = q.put_get(node._data)
        assert cnt_ == 10
        assert data is node._data
        assert node.queue_priority == 1
        node.bound = 2
        cnt_ = q.put(node._data)
        assert node.queue_priority == 2
        assert cnt_ == 11
        node2 = Node(size=0)
        node2.bound = 3
        assert node2.queue_priority is None
        cnt_, data = q.put_get(node2._data)
        assert cnt_ == 12
        assert data is node2._data
        assert node2.queue_priority == 3
        node2.bound = 1
        cnt_, data = q.put_get(node2._data)
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert data is node._data
        assert q.size() == 1

class TestCustomPriorityQueue(object):

    def test_missing_queue_priority(self):
        q = CustomPriorityQueue(minimize)
        node = Node(size=0)
        node.bound = 0
        assert node.queue_priority is None
        with pytest.raises(ValueError):
            q.put(node._data)
        with pytest.raises(ValueError):
            q.put_get(node._data)
        node.queue_priority = 1
        q.put(node._data)
        child = node.new_child()
        assert child.queue_priority is None
        with pytest.raises(ValueError):
            q.put(child._data)
        with pytest.raises(ValueError):
            q.put_get(child._data)

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
        node = Node(size=0)
        node.bound = 0
        node.queue_priority = 1
        cnt_, data = q.put_get(node._data)
        assert cnt_ == 10
        assert data is node._data
        assert node.queue_priority == 1
        assert q.bound() is None
        node.queue_priority = 2
        cnt_ = q.put(node._data)
        assert node.queue_priority == 2
        assert cnt_ == 11
        assert q.bound() == 0
        node2 = Node(size=0)
        node2.bound = 1
        node2.queue_priority = 3
        cnt_, data = q.put_get(node2._data)
        assert cnt_ == 12
        assert data is node2._data
        assert node2.queue_priority == 3
        assert q.bound() == 0
        node2.queue_priority = 1
        cnt_, data = q.put_get(node2._data)
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert data is node._data
        assert q.size() == 1
        assert q.bound() == 1

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
        node = Node(size=0)
        node.bound = 0
        node.queue_priority = 1
        cnt_, data = q.put_get(node._data)
        assert cnt_ == 10
        assert data is node._data
        assert node.queue_priority == 1
        assert q.bound() is None
        node.queue_priority = 2
        cnt_ = q.put(node._data)
        assert node.queue_priority == 2
        assert cnt_ == 11
        assert q.bound() == 0
        node2 = Node(size=0)
        node2.bound = 1
        node2.queue_priority = 3
        cnt_, data = q.put_get(node2._data)
        assert cnt_ == 12
        assert data is node2._data
        assert node2.queue_priority == 3
        assert q.bound() == 0
        node2.queue_priority = 1
        cnt_, data = q.put_get(node2._data)
        assert node2.queue_priority == 1
        assert cnt_ == 13
        assert data is node._data
        assert q.size() == 1
        assert q.bound() == 1

class TestBestObjectiveFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = BestObjectiveFirstPriorityQueue(minimize)
        node = Node(size=0)
        node.bound = -1
        assert node.queue_priority is None
        node.objective = 1
        assert q.put(node._data) == 0
        assert node.objective == 1
        assert node.queue_priority == -1
        child = node.new_child()
        assert child.objective == 1
        child.objective = 0
        assert child.queue_priority is None
        cnt, data_ = q.put_get(child._data)
        assert child.queue_priority == 0
        assert cnt == 1
        assert data_ is child._data
        child.objective = 2
        cnt, data_ = q.put_get(child._data)
        assert child.queue_priority == -2
        assert cnt == 2
        assert data_ is node._data
        assert q.bound() == -1

        q = BestObjectiveFirstPriorityQueue(maximize)
        node = Node(size=0)
        node.bound = 3
        assert node.queue_priority is None
        node.objective = 1
        assert q.put(node._data) == 0
        assert node.objective == 1
        assert node.queue_priority == 1
        child = node.new_child()
        assert child.objective == 1
        child.objective = 2
        assert child.queue_priority is None
        cnt, data_ = q.put_get(child._data)
        assert child.queue_priority == 2
        assert cnt == 1
        assert data_ is child._data
        child.objective = 0
        cnt, data_ = q.put_get(child._data)
        assert child.queue_priority == 0
        assert cnt == 2
        assert data_ is node._data
        assert q.bound() == 3

class TestBreadthFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = BreadthFirstPriorityQueue(minimize)
        node = Node(size=0)
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.tree_depth == 0
        assert node.queue_priority == 0
        child = node.new_child()
        assert child.tree_depth == 1
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority == -child.tree_depth

        l1 = Node(size=0)
        l1.bound = 1
        l2 = l1.new_child()
        l3 = l2.new_child()
        q = BreadthFirstPriorityQueue(minimize)
        q.put(l2._data)
        cnt, data_ = q.put_get(l1._data)
        assert cnt == 1
        assert data_ is l1._data
        cnt, data_ = q.put_get(l3._data)
        assert cnt == 2
        assert data_ is l2._data
        assert q.bound() == 1

class TestDepthFirstPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = DepthFirstPriorityQueue(minimize)
        node = Node(size=0)
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.tree_depth == 0
        assert node.queue_priority == 0
        child = node.new_child()
        assert child.tree_depth == 1
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority == child.tree_depth

        l1 = Node(size=0)
        l1.bound = 1
        l2 = l1.new_child()
        l3 = l2.new_child()
        q = DepthFirstPriorityQueue(minimize)
        q.put(l2._data)
        cnt, data_ = q.put_get(l3._data)
        assert cnt == 1
        assert data_ is l3._data
        cnt, data_ = q.put_get(l2._data)
        assert cnt == 2
        assert data_ is l2._data
        assert q.bound() == 1

class TestFIFOQueue(object):

    def test_overwrites_queue_priority(self):
        q = FIFOQueue(minimize)
        node = Node(size=0)
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.queue_priority == 0
        child = node.new_child()
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority == -1

        l1 = Node(size=0)
        l1.bound = 1
        l2 = l1.new_child()
        l3 = l2.new_child()
        q = FIFOQueue(minimize)
        cnt, data = q.put_get(l2._data)
        assert cnt == 0
        assert data is l2._data
        cnt = q.put(l2._data)
        assert cnt == 1
        cnt, data_ = q.put_get(l3._data)
        assert cnt == 2
        assert data_ is l2._data
        cnt, data_ = q.put_get(l1._data)
        assert cnt == 3
        assert data_ is l3._data
        assert q.bound() == 1

class TestRandomPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = RandomPriorityQueue(minimize)
        node = Node(size=0)
        node.bound = 0
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.queue_priority is not None
        assert 0 <= node.queue_priority <= 1
        child = node.new_child()
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority is not None
        assert 0 <= child.queue_priority <= 1

        l1 = Node(size=0)
        l1.bound = 1
        l2 = l1.new_child()
        l3 = l2.new_child()
        q = RandomPriorityQueue(minimize)
        assert l2.queue_priority is None
        cnt, data = q.put_get(l2._data)
        assert data is l2._data
        assert l2.queue_priority is not None
        assert 0 <= l2.queue_priority <= 1
        assert cnt == 0
        cnt = q.put(l2._data)
        assert cnt == 1
        assert l3.queue_priority is None
        cnt, data_ = q.put_get(l3._data)
        assert cnt == 2
        assert l3.queue_priority is not None
        assert 0 <= l3.queue_priority <= 1
        assert data_ is max([l2, l3],
                            key=lambda x_: x_.queue_priority)._data


class TestLocalGapPriorityQueue(object):

    def test_overwrites_queue_priority(self):
        q = LocalGapPriorityQueue(minimize)
        node = Node(size=0)
        node.bound = -inf
        node.objective = inf
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.queue_priority is not None
        assert node.queue_priority == inf
        child = node.new_child()
        assert child.bound == -inf
        assert child.objective == inf
        child.bound = 0
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority is not None
        assert child.queue_priority == inf
        child = child.new_child()
        assert child.bound == 0
        assert child.objective == inf
        child.objective = 1
        assert child.queue_priority is None
        assert q.put(child._data) == 2
        assert child.queue_priority is not None
        assert child.queue_priority == 1

        l1 = Node(size=0)
        l1.bound = 1
        l1.objective = 5
        l2 = l1.new_child()
        l3 = l2.new_child()
        q = LocalGapPriorityQueue(minimize)
        assert l2.queue_priority is None
        cnt, data = q.put_get(l2._data)
        assert data is l2._data
        assert l2.queue_priority is not None
        assert l2.queue_priority == 4
        assert cnt == 0
        cnt = q.put(l2._data)
        assert cnt == 1
        assert l3.queue_priority is None
        l3.objective = 6
        cnt, data_ = q.put_get(l3._data)
        assert cnt == 2
        assert l3.queue_priority is not None
        assert l3.queue_priority == 5
        assert data_ is l3._data

        q = LocalGapPriorityQueue(maximize)
        node = Node(size=0)
        node.bound = inf
        node.objective = -inf
        assert node.queue_priority is None
        assert q.put(node._data) == 0
        assert node.queue_priority is not None
        assert node.queue_priority == inf
        child = node.new_child()
        assert child.bound == inf
        assert child.objective == -inf
        child.bound = 0
        assert child.queue_priority is None
        assert q.put(child._data) == 1
        assert child.queue_priority is not None
        assert child.queue_priority == inf
        child = child.new_child()
        assert child.bound == 0
        assert child.objective == -inf
        child.objective = -1
        assert child.queue_priority is None
        assert q.put(child._data) == 2
        assert child.queue_priority is not None
        assert child.queue_priority == 1

        l1 = Node(size=0)
        l1.bound = -1
        l1.objective = -5
        l2 = l1.new_child()
        l3 = l2.new_child()
        q = LocalGapPriorityQueue(maximize)
        assert l2.queue_priority is None
        cnt, data = q.put_get(l2._data)
        assert data is l2._data
        assert l2.queue_priority is not None
        assert l2.queue_priority == 4
        assert cnt == 0
        cnt = q.put(l2._data)
        assert cnt == 1
        assert l3.queue_priority is None
        l3.objective = -6
        cnt, data_ = q.put_get(l3._data)
        assert cnt == 2
        assert l3.queue_priority is not None
        assert l3.queue_priority == 5
        assert data_ is l3._data
