import itertools

import pytest
from runtests.mpi import MPITest

import six
from six.moves import zip

import pybnb

from .common import mpi_available

def left_child(i):
    return 2*i + 1

def right_child(i):
    return 2*i + 2

def log2floor(n):
    assert n > 0
    return n.bit_length() - 1

def height(size):
    return log2floor(size)

def set_none(heap, i):
    if i < len(heap):
        heap[i] = None
        set_none(heap, left_child(i))
        set_none(heap, right_child(i))

def set_one(heap, i):
    if i < len(heap):
        if heap[i] is not None:
            heap[i] = 1
            set_one(heap, left_child(i))
            set_one(heap, right_child(i))

def is_terminal(heap, i):
    N = len(heap)
    c1 = left_child(i)
    c2 = right_child(i)
    return ((c1 >= N) or (heap[c1] is None)) and \
           ((c2 >= N) or (heap[c2] is None))

def get_bound(heap, how=min):
    N = len(heap)
    assert N >= 1
    assert heap[0] is not None
    terminal_bounds = []
    for i in range(N):
        if (heap[i] is not None) and is_terminal(heap, i):
            terminal_bounds.append(heap[i])
    return how(terminal_bounds)

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def gen_heaps(k):
    for heap_size in range(1,2**(k+1)):
        h = height(heap_size)
        for level_none in range(h,h+1):
            level_nodes = list(filter(lambda x: x < heap_size,
                                      range((2**level_none)-1,(2**(level_none+1))-1)))
            for none_list in sorted(powerset(level_nodes)):
                if len(none_list) == len(level_nodes):
                    continue
                heap_master = [0]*heap_size
                for i in none_list:
                    set_none(heap_master, i)
                for level in range(0,h+1):
                    nodes = filter(lambda x: (x < heap_size) and (heap_master[x] is not None),
                                   range((2**level)-1,(2**(level+1))-1))
                    for nodes_list in sorted(powerset(nodes)):
                        if (len(nodes_list) == 0) and \
                           level != 0:
                            continue
                        heap = [0]*heap_size
                        for i in none_list:
                            set_none(heap, i)
                        for i in nodes_list:
                            set_one(heap, i)
                        yield heap

class DiscreteMin(pybnb.Problem):

    def __init__(self,
                 objectives,
                 bound_bheap,
                 default_objective):
        assert len(bound_bheap) >= 1
        self._objectives = objectives
        self._bound_bheap = bound_bheap
        self._default_objective = default_objective
        self._node = None

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.minimize

    def objective(self):
        assert self._node is not None
        tree_id = self._node.tree_id
        tree_id is not None
        return self._objectives.get(tree_id,
                                    self._default_objective)

    def bound(self):
        assert self._node is not None
        tree_id = self._node.tree_id
        tree_id is not None
        tree_id >= 0
        return self._bound_bheap[tree_id]

    def save_state(self, node):
        node.resize(0)

    def load_state(self, node):
        self._node = node

    def branch(self, parent):
        i = parent.tree_id
        assert i >= 0
        assert i < len(self._bound_bheap)
        left_tree_id =  2*i + 1
        children = []
        if (left_tree_id < len(self._bound_bheap)) and \
           (self._bound_bheap[left_tree_id] is not None):
            child = parent.new_children(1)[0]
            child.tree_id = left_tree_id
            children.append(child)
        right_tree_id = 2*i + 2
        if (right_tree_id < len(self._bound_bheap)) and \
           (self._bound_bheap[right_tree_id] is not None):
            child = parent.new_children(1)[0]
            child.tree_id = right_tree_id
            children.append(child)
        return children

def _test_heaps(comm):
    solver = pybnb.Solver(comm=comm)
    if solver.comm is None:
        pass
    else:
        if solver.comm.rank == 0:
            pass
        elif solver.comm.rank == 1:
            pass
        elif solver.comm.rank == 2:
            pass
    for heap in gen_heaps(2):
        get_bound(heap)
        node_list = [None, len(heap)] + [i for i in range(len(heap))
                                         if heap[i] is not None]
        """
        for default_objective in [None, 2]:
            for objective_node in node_list:
                if objective_node is not None:
                    problem = DiscreteMin({objective_node: 1},
                                          heap,
                                          default_objective=2)
                else:
                    problem = DiscreteMin({},
                                          heap,
                                          default_objective=1)
                results = solver.solve(problem, log=None)
                if objective_node == len(heap):
                    assert results.objective == 2
                else:
                    assert results.objective == 1
                if results.bound != get_bound(heap):   #pragma:nocover
                    # bad
                    if (solver.comm is None) or \
                       (solver.comm.rank == 0):
                        print(heap)
                        print(results.bound)
                        print(get_bound(heap))
                        print(objective_node)
                    if solver.comm is not None:
                        solver.comm.Barrier()
                    assert False
        """
def test_heaps_nocomm():
    _test_heaps(None)

if mpi_available:

    @MPITest(commsize=[1, 2, 3])
    def test_heaps_comm(comm):
        _test_heaps(comm)
