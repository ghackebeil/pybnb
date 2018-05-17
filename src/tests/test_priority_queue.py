import random

import pytest

from pybnb.priority_queue import \
    _NoThreadingMaxPriorityFirstQueue

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
        assert q.next() is None
        for i in range(10):
            q.put(i,0)
            assert q.size() == i+1
            assert_isheap(q._heap)
        for i in range(10):
            x_ = q.next()
            x = q.get()
            assert x_ is x
            assert x == i
            assert q.size() == 10-(i+1)
            assert_isheap(q._heap)
        assert q.size() == 0
        assert q.get() is None
        assert q.next() is None
        assert_isheap(q._heap)

    def test_put_get(self):
        q = _NoThreadingMaxPriorityFirstQueue()
        with pytest.raises(AssertionError):
            q.put(None, 0)
        assert sorted(q.items()) == []
        assert_isheap(q._heap)
        q.put(1,1)
        assert sorted(q.items()) == [1]
        assert_isheap(q._heap)
        q.put(2,2)
        assert sorted(q.items()) == [1,2]
        assert_isheap(q._heap)
        q.put(3,2)
        assert sorted(q.items()) == [1,2,3]
        assert_isheap(q._heap)
        q.put(4,-4)
        assert sorted(q.items()) == [1,2,3,4]
        assert_isheap(q._heap)
        assert q.next() == 2
        assert q.get() == 2
        assert sorted(q.items()) == [1,3,4]
        assert_isheap(q._heap)
        assert q.next() == 3
        assert q.get() == 3
        assert sorted(q.items()) == [1,4]
        assert_isheap(q._heap)
        assert q.next() == 1
        assert q.get() == 1
        assert sorted(q.items()) == [4]
        assert_isheap(q._heap)
        assert q.next() == 4
        assert q.get() == 4
        assert sorted(q.items()) == []
        assert_isheap(q._heap)
        assert q.next() is None

    def test_filter(self):
        for k in range(-10,11):
            cutoff = k*11
            def _filter(priority, item):
                assert priority == 1
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
