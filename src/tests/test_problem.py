import time

from pybnb.common import (minimize,
                          maximize,
                          inf)
from pybnb.problem import (Problem,
                           _SolveInfo,
                           _SimpleSolveInfoCollector)
import pytest

class MinProblem(Problem):
    def sense(self): return minimize

class MaxProblem(Problem):
    def sense(self): return maximize

class TestProblem(object):

    def test_infeasible_objective(self):
        p = MinProblem()
        assert p.infeasible_objective() == inf
        p = MaxProblem()
        assert p.infeasible_objective() == -inf

    def test_unbounded_objective(self):
        p = MinProblem()
        assert p.unbounded_objective() == -inf
        p = MaxProblem()
        assert p.unbounded_objective() == inf

    def test_default_methods(self):
        p = Problem()
        with pytest.raises(NotImplementedError):
            p.sense()
        with pytest.raises(NotImplementedError):
            p.objective()
        with pytest.raises(NotImplementedError):
            p.load_state(None)
        with pytest.raises(NotImplementedError):
            p.save_state(None)
        with pytest.raises(NotImplementedError):
            p.branch(None)
        p.notify_solve_begins(None, None, None)
        p.notify_new_best_objective_received(None)
        p.notify_new_best_objective(None)
        p.notify_solve_finished(None, None, None)

class Test_SolveInfo(object):

    def test_methods(self):
        info = _SolveInfo()
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.explored_nodes_count += 1
        assert info.explored_nodes_count == 1
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info._increment_explored_nodes_stat(2)
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.total_queue_time += 1.5
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 1.5
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.queue_call_count += 1
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 1.5
        assert info.queue_call_count == 1
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info._increment_queue_stat(2.0, 2)
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.total_objective_time += 1.5
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 1.5
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.objective_call_count += 1
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 1.5
        assert info.objective_call_count == 1
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info._increment_objective_stat(2.0, 2)
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.total_bound_time += 1.5
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 1.5
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.bound_call_count += 1
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 1.5
        assert info.bound_call_count == 1
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info._increment_bound_stat(2.0, 2)
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 3.5
        assert info.bound_call_count == 3
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.total_branch_time += 1.5
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 3.5
        assert info.bound_call_count == 3
        assert info.total_branch_time == 1.5
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.branch_call_count += 1
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 3.5
        assert info.bound_call_count == 3
        assert info.total_branch_time == 1.5
        assert info.branch_call_count == 1
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info._increment_branch_stat(2.0, 2)
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 3.5
        assert info.bound_call_count == 3
        assert info.total_branch_time == 3.5
        assert info.branch_call_count == 3
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        info.total_load_state_time += 1.5
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 3.5
        assert info.bound_call_count == 3
        assert info.total_branch_time == 3.5
        assert info.branch_call_count == 3
        assert info.total_load_state_time == 1.5
        assert info.load_state_call_count == 0
        info.load_state_call_count += 1
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 3.5
        assert info.bound_call_count == 3
        assert info.total_branch_time == 3.5
        assert info.branch_call_count == 3
        assert info.total_load_state_time == 1.5
        assert info.load_state_call_count == 1
        info._increment_load_state_stat(2.0, 2)
        assert info.explored_nodes_count == 3
        assert info.total_queue_time == 3.5
        assert info.queue_call_count == 3
        assert info.total_objective_time == 3.5
        assert info.objective_call_count == 3
        assert info.total_bound_time == 3.5
        assert info.bound_call_count == 3
        assert info.total_branch_time == 3.5
        assert info.branch_call_count == 3
        assert info.total_load_state_time == 3.5
        assert info.load_state_call_count == 3
        info.reset()
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0

    def test_add_from(self):
        info = _SolveInfo()
        info.explored_nodes_count = 1
        info.total_queue_time = 2
        info.queue_call_count = 3
        info.total_objective_time = 4
        info.objective_call_count = 5
        info.total_bound_time = 6
        info.bound_call_count = 7
        info.total_branch_time = 8
        info.branch_call_count = 9
        info.total_load_state_time = 10
        info.load_state_call_count = 11
        info1 = _SolveInfo()
        info1.explored_nodes_count = -2
        info1.total_queue_time = -3
        info1.queue_call_count = -4
        info1.total_objective_time = -5
        info1.objective_call_count = -6
        info1.total_bound_time = -7
        info1.bound_call_count = -8
        info1.total_branch_time = -9
        info1.branch_call_count = -10
        info1.total_load_state_time = -11
        info1.load_state_call_count = -12
        info.add_from(info1)
        assert info.explored_nodes_count  == -1
        assert info.total_queue_time  == -1
        assert info.queue_call_count  == -1
        assert info.total_objective_time  == -1
        assert info.objective_call_count  == -1
        assert info.total_bound_time  == -1
        assert info.bound_call_count  == -1
        assert info.total_branch_time  == -1
        assert info.branch_call_count  == -1
        assert info.total_load_state_time  == -1
        assert info.load_state_call_count  == -1
        with pytest.raises(TypeError):
            info.add_from(1)

    def test__SimpleSolveInfoCollector(self):
        class Junk(Problem):
            def __init__(self):
                self.d = {}
                self.d['sense'] = False
                self.d['objective'] = False
                self.d['bound'] = False
                self.d['save_state'] = False
                self.d['load_state'] = False
                self.d['branch'] = False
                self.d['notify_solve_begins'] = False
                self.d['notify_new_best_objective_received'] = False
                self.d['notify_new_best_objective'] = False
                self.d['notify_solve_finished'] = False
            def sense(self):
                self.d['sense'] = True
                return -1
            def objective(self):
                time.sleep(0.01)
                self.d['objective'] = True
                return -2
            def bound(self):
                time.sleep(0.01)
                self.d['bound'] = True
                return -3
            def save_state(self, node):
                self.d['save_state'] = True
            def load_state(self, node):
                time.sleep(0.01)
                self.d['load_state'] = True
            def branch(self, parent_node):
                time.sleep(0.01)
                self.d['branch'] = True
                return ()
            def notify_solve_begins(self,
                                      comm,
                                      worker_comm,
                                      convergence_checker):
                self.d['notify_solve_begins'] = True
            def notify_new_best_objective_received(self,
                                                   objective):
                self.d['notify_new_best_objective_received'] = True
            def notify_new_best_objective(self,
                                          objective):
                self.d['notify_new_best_objective'] = True
            def notify_solve_finished(self,
                                      comm,
                                      worker_comm,
                                      results):
                self.d['notify_solve_finished'] = True
        j = Junk()
        assert j.d['sense'] == False
        assert j.d['objective'] == False
        assert j.d['bound'] == False
        assert j.d['save_state'] == False
        assert j.d['load_state'] == False
        assert j.d['branch'] == False
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        p = _SimpleSolveInfoCollector(j)
        p.set_clock(time.time)
        info = _SolveInfo()
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        p.set_solve_info_object(info)
        assert j.d['sense'] == False
        assert j.d['objective'] == False
        assert j.d['bound'] == False
        assert j.d['save_state'] == False
        assert j.d['load_state'] == False
        assert j.d['branch'] == False
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        p.sense()
        assert j.d['sense'] == True
        assert j.d['objective'] == False
        assert j.d['bound'] == False
        assert j.d['save_state'] == False
        assert j.d['load_state'] == False
        assert j.d['branch'] == False
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time == 0
        assert info.objective_call_count == 0
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        p.objective()
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == False
        assert j.d['save_state'] == False
        assert j.d['load_state'] == False
        assert j.d['branch'] == False
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time == 0
        assert info.bound_call_count == 0
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        p.bound()
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == False
        assert j.d['load_state'] == False
        assert j.d['branch'] == False
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        p.save_state(None)
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == True
        assert j.d['load_state'] == False
        assert j.d['branch'] == False
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time == 0
        assert info.load_state_call_count == 0
        p.load_state(None)
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == True
        assert j.d['load_state'] == True
        assert j.d['branch'] == False
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time == 0
        assert info.branch_call_count == 0
        assert info.total_load_state_time > 0
        assert info.load_state_call_count == 1
        list(p.branch(None))
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == True
        assert j.d['load_state'] == True
        assert j.d['branch'] == True
        assert j.d['notify_solve_begins'] == False
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time > 0
        assert info.branch_call_count == 1
        assert info.total_load_state_time > 0
        assert info.load_state_call_count == 1
        p.notify_solve_begins(None, None, None)
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == True
        assert j.d['load_state'] == True
        assert j.d['branch'] == True
        assert j.d['notify_solve_begins'] == True
        assert j.d['notify_new_best_objective_received'] == False
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time > 0
        assert info.branch_call_count == 1
        assert info.total_load_state_time > 0
        assert info.load_state_call_count == 1
        p.notify_new_best_objective_received(None)
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == True
        assert j.d['load_state'] == True
        assert j.d['branch'] == True
        assert j.d['notify_solve_begins'] == True
        assert j.d['notify_new_best_objective_received'] == True
        assert j.d['notify_new_best_objective'] == False
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time > 0
        assert info.branch_call_count == 1
        assert info.total_load_state_time > 0
        assert info.load_state_call_count == 1
        p.notify_new_best_objective(None)
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == True
        assert j.d['load_state'] == True
        assert j.d['branch'] == True
        assert j.d['notify_solve_begins'] == True
        assert j.d['notify_new_best_objective_received'] == True
        assert j.d['notify_new_best_objective'] == True
        assert j.d['notify_solve_finished'] == False
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time > 0
        assert info.branch_call_count == 1
        assert info.total_load_state_time > 0
        assert info.load_state_call_count == 1
        p.notify_solve_finished(None, None, None)
        assert j.d['sense'] == True
        assert j.d['objective'] == True
        assert j.d['bound'] == True
        assert j.d['save_state'] == True
        assert j.d['load_state'] == True
        assert j.d['branch'] == True
        assert j.d['notify_solve_begins'] == True
        assert j.d['notify_new_best_objective_received'] == True
        assert j.d['notify_new_best_objective'] == True
        assert j.d['notify_solve_finished'] == True
        assert info.explored_nodes_count == 0
        assert info.total_queue_time == 0
        assert info.queue_call_count == 0
        assert info.total_objective_time > 0
        assert info.objective_call_count == 1
        assert info.total_bound_time > 0
        assert info.bound_call_count == 1
        assert info.total_branch_time > 0
        assert info.branch_call_count == 1
        assert info.total_load_state_time > 0
        assert info.load_state_call_count == 1
