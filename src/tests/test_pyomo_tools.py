import pytest

import pybnb

pyomo_available = False
try:
    import pyomo.kernel as pmo
    from pybnb.pyomo_tools import (_add_tmp_component,
                                   _create_optimality_bound)
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
