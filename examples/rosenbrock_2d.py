import pybnb

pyomo_available = False
try:
    import pyomo.kernel as pmo
    if getattr(pmo,'version_info',(0,)*3) >= (5,4,3):  #pragma:nocover
        pyomo_available = True
except ImportError:                                    #pragma:nocover
    pass
if not pyomo_available:                                #pragma:nocover
    raise ImportError("This example requires Pyomo 5.4.3 or above")
else:
    from pybnb.pyomo_tools import PyomoProblem

class McCormickEnvelope(pmo.constraint_tuple):
    """A class that stores constraints defining
    the convex envelope for the function: z = xy
    """
    def __init__(self, x, y, z):
        assert x.has_lb() and x.has_ub()
        assert y.has_lb() and y.has_ub()
        self.x = x
        self.y = y
        self.z = z
        super(McCormickEnvelope, self).__init__(
            (pmo.constraint(),
             pmo.constraint(),
             pmo.constraint(),
             pmo.constraint()))
        self.update_constraints()

    def update_constraints(self):
        """Rebuild constraints using current domain variable bounds"""
        x, y, z = self.x, self.y, self.z
        assert x.has_lb() and x.has_ub()
        assert y.has_lb() and y.has_ub()
        self[0].body = z - y.lb*x - x.lb*y
        self[0].lb = -x.lb*y.lb
        self[1].body = z - y.ub*x - x.ub*y
        self[1].lb = -x.ub*y.ub
        self[2].body = z - y.ub*x - x.lb*y
        self[2].ub = -x.lb*y.ub
        self[3].body = z - y.lb*x - x.ub*y
        self[3].ub = -x.ub*y.lb

class SquaredEnvelope(pmo.constraint_tuple):
    """A class that stores constraints defining
    the convex envelope for the function: z = x^2"""

    def __init__(self, x, z):
        assert x.has_lb() and x.has_ub()
        self.x = x
        self.z = z
        super(SquaredEnvelope, self).__init__(
            (pmo.constraint(),
             pmo.constraint()))
        self[0].body = self.z - self.x**2
        self[0].lb = 0
        self.update_constraints()

    def update_constraints(self):
        """Rebuild constraints using current domain variable bounds"""
        x, z = self.x, self.z
        assert x.has_lb() and x.has_ub()
        self[1].body = z - (x.lb + x.ub)*x
        self[1].ub = -x.lb*x.ub

    def derived_output_bounds(self):
        x = self.x
        assert x.has_lb() and x.has_ub()
        vals = (x.lb**2, x.ub**2)
        if (x.lb <= 0) and (x.ub >= 0):
            lb = 0.0
        else:
            lb = min(vals)
        ub = max(vals)
        return lb, ub

class Rosenbrock2D(PyomoProblem):

    def __init__(self, xL, xU, yL, yU, branch_abstol=1e-3):
        assert branch_abstol > 0
        assert xL <= xU
        assert yL <= yU
        self._branch_abstol = branch_abstol
        self._model = pmo.block()
        x = self._model.x = pmo.variable(lb=xL, ub=xU)
        y = self._model.y = pmo.variable(lb=yL, ub=yU)
        x2 = self._model.x2 = pmo.variable(
            lb=-pybnb.inf,
            ub=pybnb.inf)
        x2y = self._model.x2y = pmo.variable(
            lb=-pybnb.inf,
            ub=pybnb.inf)
        self._model.x2_c = SquaredEnvelope(x, x2)
        # Temporarily provide bounds to the x2 variable so
        # they can be used to build the McCormick
        # constraints for x2y. After that, they are no
        # longer needed as they are enforced by the
        # McCormickEnvelope constraints.
        x2.bounds = self._model.x2_c.derived_output_bounds()
        self._model.x2y_c = McCormickEnvelope(x2, y, x2y)
        x2.bounds = (-pybnb.inf,
                     pybnb.inf)

        # original objective
        self._model.f = pmo.expression(
            (x**4 - 2*(x**2)*y + \
             0.5*(x**2) - x + \
             (y**2) + 0.5))
        # convex relaxation
        self._model.f_convex = pmo.expression(
            (x**4 - 2*x2y + \
             0.5*(x**2) - x + \
             (y**2) + 0.5))
        self._model.objective = pmo.objective(sense=pmo.minimize)
        self._ipopt = pmo.SolverFactory("ipopt")
        self._ipopt.options['tol'] = 1e-9
        self._last_bound_was_feasible = False

        # make sure the PyomoProblem initializer is called
        # after the model is built
        super(Rosenbrock2D, self).__init__()

    def check_feasible(self):
        """Check if the currently loaded solution
        is feasible for the true model."""
        # assumes the relaxed model was just solved
        assert not self._model.x2y.stale
        x = self._model.x
        y = self._model.y
        resid = 0.0
        # x.lb <= x <= x.ub
        resid += min(0,x.slack)**2
        # y.lb <= y <= y.ub
        resid += min(0,y.slack)**2
        # x2y <= y(x^2)
        resid += min(0, (x.value*x.value*y.value) - \
                        self._model.x2y.value)**2
        resid = resid**(0.5)
        if resid < 1e-5:
            return True
        else:
            return False

    def setup_model_for_objective(self):
        self._model.objective.expr = self._model.f
        self._model.x2_c.deactivate()
        self._model.x2y_c.deactivate()

    def setup_model_for_bound(self):
        self._model.objective.expr = self._model.f_convex
        self._model.x2_c.activate()
        self._model.x2y_c.activate()

    def rebuild_convex_envelopes(self):
        self._model.x2_c.update_constraints()
        # Temporarily provide bounds to the x2 variable so
        # they can be used to build the McCormick
        # constraints for x2y. After that, they are no
        # longer needed as they are enforced by the
        # SquaredEnvelope constraints.
        self._model.x2.bounds = self._model.x2_c.derived_output_bounds()
        self._model.x2y_c.update_constraints()
        self._model.x2.bounds = (-pybnb.inf,
                                 pybnb.inf)

    #
    # Implement PyomoProblem abstract methods
    #

    @property
    def pyomo_model(self):
        return self._model

    @property
    def pyomo_model_objective(self):
        return self._model.objective

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.minimize

    def objective(self):
        self.setup_model_for_objective()
        results = self._ipopt.solve(self._model, load_solutions=False)
        if (str(results.solver.status) == "ok") and \
           (str(results.solver.termination_condition) == "optimal"):
            self._model.load_solution(results.solution(0))
            assert self._model.x2y.stale
            return round(self._model.objective(), 7)
        else:
            assert str(results.solver.status) == "warning"
            if str(results.solver.termination_condition) == "unbounded":
                return self.unbounded_objective()
            else:
                assert str(results.solver.termination_condition) == "infeasible"
                return self.infeasible_objective()

    def bound(self):
        self.setup_model_for_bound()
        self._last_bound_was_feasible = False
        results = self._ipopt.solve(self._model, load_solutions=False)
        if (str(results.solver.status) == "ok") and \
           (str(results.solver.termination_condition) == "optimal"):
            self._model.load_solution(results.solution(0))
            self._last_bound_was_feasible = self.check_feasible()
            return round(self._model.objective(), 7)
        else:
            assert str(results.solver.status) == "warning"
            if str(results.solver.termination_condition) == "unbounded":
                return self.unbounded_objective()
            else:
                assert str(results.solver.termination_condition) == "infeasible"
                return self.infeasible_objective()

    def save_state(self, node):
        node.resize(4)
        state = node.state
        state[0] = self._model.x.lb
        state[1] = self._model.x.ub
        state[2] = self._model.y.lb
        state[3] = self._model.y.ub

    def load_state(self, node):
        state = node.state
        assert len(state) == 4
        self._model.x.lb = float(state[0])
        self._model.x.ub = float(state[1])
        self._model.y.lb = float(state[2])
        self._model.y.ub = float(state[3])
        self.rebuild_convex_envelopes()
        self._last_bound_was_feasible = False

    def branch(self, parent):
        if self._last_bound_was_feasible:
            return ()
        xL, xU = self._model.x.bounds
        yL, yU = self._model.y.bounds
        xdist = float(xU-xL)
        ydist = float(yU-yL)
        branch_var = None
        if xdist > ydist:
            branch_var = self._model.x
            dist = xdist
            L = xL
            U = xU
        else:
            branch_var = self._model.y
            dist = ydist
            L = yL
            U = yU
        if dist/2.0 < self._branch_abstol:
            return ()
        # branch
        mid = 0.5*(L + U)
        left = parent.new_child()
        branch_var.bounds = (L, mid)
        self.save_state(left)
        right = parent.new_child()
        branch_var.bounds = (mid, U)
        self.save_state(right)
        # reset the variable bounds
        branch_var.bounds = (L, U)
        return [left, right]

if __name__ == "__main__":
    import pybnb.misc

    problem = Rosenbrock2D(xL=-25, xU=25,
                           yL=-25 ,yU=25)
    pybnb.misc.create_command_line_solver(problem)
