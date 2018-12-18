import pytest

import pybnb

pyomo_available = False
try:
    import pyomo.kernel as pmo
    from pybnb.pyomo_tools import (_add_tmp_component,
                                   _create_optimality_bound,
                                   PyomoProblem,
                                   RangeReductionProblem)
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

    def test_add_tmp_component(self):
        model = pmo.block()
        obj = pmo.objective()
        name = _add_tmp_component(model,
                                  "objective",
                                  obj)
        assert name == "objective"
        assert getattr(model, name) is obj
        obj = pmo.objective()
        name = _add_tmp_component(model,
                                  "objective",
                                  obj)
        assert name == ".objective."
        assert getattr(model, name) is obj

    def test_create_optimality_bound(self):
        # max
        problem = MaxProblem()
        model = pmo.block()
        model.objective = pmo.objective(sense=pmo.maximize)
        con = _create_optimality_bound(problem,
                                       model.objective,
                                       100)
        assert con.ctype is pmo.constraint._ctype
        assert con.body is model.objective
        assert con.lb == 100
        # min
        problem = MinProblem()
        model = pmo.block()
        model.objective = pmo.objective(sense=pmo.minimize)
        con = _create_optimality_bound(problem,
                                       model.objective,
                                       100)
        assert con.ctype is pmo.constraint._ctype
        assert con.body is model.objective
        assert con.ub == 100

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

        rr_junk = RangeReductionProblem(junk, pybnb.inf)
        assert rr_junk.objective() == pybnb.inf
        rr_junk.notify_new_best_objective_received(None, 1)
        assert rr_junk.objective() == 1
        rr_junk.notify_new_best_objective(None, 2)
        assert rr_junk.objective() == 2
