import pytest

import pybnb

pyomo_available = False
try:
    import pyomo.kernel as pmo
    from pybnb.pyomo.misc import (add_tmp_component,
                                  create_optimality_bound,
                                  correct_integer_lb,
                                  correct_integer_ub)
    from pybnb.pyomo.problem import PyomoProblem
    from pybnb.pyomo.range_reduction import RangeReductionProblem
    pyomo_available = True
except:
    pass

class MinProblem(pybnb.Problem):
    def sense(self): return pybnb.minimize

class MaxProblem(pybnb.Problem):
    def sense(self): return pybnb.maximize

@pytest.mark.skipif(not pyomo_available,
                    reason="Pyomo is not available")
class Test(object):

    def test_correct_integer_lb(self):
        # it is important that we use floating point numbers
        # that are exactly representable in base 2 so the
        # edge cases behave as expected.
        eps = 0.03125
        tol = 0.25
        for b in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            assert correct_integer_lb(b-tol-eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b-tol,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b-tol+eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b-eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b+eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b+tol-eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b+tol,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_lb(b+tol+eps,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_lb(b+1-eps,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_lb(b+1,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_lb(b+1+eps,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_lb(b+1+tol-eps,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_lb(b+1+tol,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_lb(b+1+tol+eps,
                                      integer_tolerance=tol) \
                == b+2

    def test_correct_integer_ub(self):
        # it is important that we use floating point numbers
        # that are exactly representable in base 2 so the
        # edge cases behave as expected.
        eps = 0.03125
        tol = 0.25
        for b in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            assert correct_integer_ub(b-tol-eps,
                                      integer_tolerance=tol) \
                == b-1
            assert correct_integer_ub(b-tol,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b-tol+eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b-eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b+eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b+tol-eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b+tol,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b+tol+eps,
                                      integer_tolerance=tol) \
                == b
            assert correct_integer_ub(b+1-eps,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_ub(b+1,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_ub(b+1+eps,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_ub(b+1+tol-eps,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_ub(b+1+tol,
                                      integer_tolerance=tol) \
                == b+1
            assert correct_integer_ub(b+1+tol+eps,
                                      integer_tolerance=tol) \
                == b+1

    def test_add_tmp_component(self):
        model = pmo.block()
        obj = pmo.objective()
        name = add_tmp_component(model,
                                 "objective",
                                 obj)
        assert name == "objective"
        assert getattr(model, name) is obj
        obj = pmo.objective()
        name = add_tmp_component(model,
                                 "objective",
                                 obj)
        assert name == ".objective."
        assert getattr(model, name) is obj

    def test_create_optimality_bound(self):
        # max
        problem = MaxProblem()
        model = pmo.block()
        model.objective = pmo.objective(sense=pmo.maximize)
        con = create_optimality_bound(problem,
                                      model.objective,
                                      100)
        assert con.ctype is pmo.constraint._ctype
        assert con.body is model.objective
        assert con.lb == 100
        # min
        problem = MinProblem()
        model = pmo.block()
        model.objective = pmo.objective(sense=pmo.minimize)
        con = create_optimality_bound(problem,
                                      model.objective,
                                      100)
        assert con.ctype is pmo.constraint._ctype
        assert con.body is model.objective
        assert con.ub == 100

    def test_sense(self):
        class Max(PyomoProblem):
            def __init__(self):
                self._pyomo_model = pmo.block()
                self._pyomo_model.o = pmo.objective(
                    sense=pmo.maximize)
                self._pyomo_model_objective = self._pyomo_model.o
                super(Max, self).__init__()
            @property
            def pyomo_model(self):
                return self._pyomo_model
            @property
            def pyomo_model_objective(self):
                return self._pyomo_model_objective
        assert Max().sense() == pybnb.maximize
        class Min(PyomoProblem):
            def __init__(self):
                self._pyomo_model = pmo.block()
                self._pyomo_model.o = pmo.objective(
                    sense=pmo.minimize)
                self._pyomo_model_objective = self._pyomo_model.o
                super(Min, self).__init__()
            @property
            def pyomo_model(self):
                return self._pyomo_model
            @property
            def pyomo_model_objective(self):
                return self._pyomo_model_objective
        assert Min().sense() == pybnb.minimize

    def test_RangeReductionProblem(self):
        class Junk(PyomoProblem):
            def __init__(self):
                self._pyomo_model = pmo.block()
                self._pyomo_model.x = pmo.variable()
                self._pyomo_model.c = pmo.constraint()
                self._pyomo_model.o = pmo.objective()
                self._pyomo_model_objective = self._pyomo_model.o
                super(Junk, self).__init__()
            @property
            def pyomo_model(self):
                return self._pyomo_model
            @property
            def pyomo_model_objective(self):
                return self._pyomo_model_objective
        junk = Junk()
        assert junk.pyomo_model_objective is junk.pyomo_model.o
        assert junk.pyomo_object_to_cid[junk.pyomo_model] == ()
        assert junk.pyomo_object_to_cid[junk.pyomo_model.x] == ('x',)
        assert junk.pyomo_object_to_cid[junk.pyomo_model.c] == ('c',)
        assert junk.pyomo_object_to_cid[junk.pyomo_model.o] == ('o',)
        assert junk.cid_to_pyomo_object[()] is junk.pyomo_model
        assert junk.cid_to_pyomo_object[('x',)] is junk.pyomo_model.x
        assert junk.cid_to_pyomo_object[('c',)] is junk.pyomo_model.c
        assert junk.cid_to_pyomo_object[('o',)] is junk.pyomo_model.o
        junk.pyomo_model.r = pmo.constraint()
        junk.update_pyomo_object_cids()
        assert junk.pyomo_object_to_cid[junk.pyomo_model.r] == ('r',)
        assert junk.cid_to_pyomo_object[('r',)] is junk.pyomo_model.r

        rr_junk = RangeReductionProblem(junk)
        assert rr_junk._best_objective == pybnb.inf
        node_ = pybnb.Node()
        node_.objective = 1
        rr_junk.notify_new_best_node(node_, False)
        assert rr_junk._best_objective == 1
        node_.objective = 2
        rr_junk.notify_new_best_node(node_, False)
        assert rr_junk._best_objective == 2
