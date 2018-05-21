#
# This example solves the bin packing problem. A custom
# heuristic is implemented to determine an integer-feasible
# objective value for each region explored. The objective
# heuristic is non-deterministic. Thus, this example may
# behave differently from run to run. Valid lower bounds are
# obtained by solving the linear relaxation using Pyomo.
#
# Pyomo version 5.4.3 or later is required to run this
# example.
#

import random

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

class BinPacking(pybnb.Problem):

    def __init__(self, V, W,
                 pyomo_solver="ipopt",
                 pyomo_solver_io="nl"):
        assert V > 0
        self.V = V
        self.W = W
        N = range(len(self.W))
        m = self.model = pmo.block()
        x = m.x = pmo.variable_dict()
        y = m.y = pmo.variable_dict()
        for i in N:
            y[i] = pmo.variable(domain=pmo.Binary)
            for j in N:
                x[i,j] = pmo.variable(domain=pmo.Binary)

        m.B = pmo.expression(sum(y.values()))
        m.objective = pmo.objective(m.B, sense=pmo.minimize)

        m.B_nontrivial = pmo.constraint(m.B >= 1)

        m.capacity = pmo.constraint_dict()
        for i in N:
            m.capacity[i] = pmo.constraint(
                sum(x[i,j]*self.W[j] for j in N) <= self.V*y[i])

        m.assign_1 = pmo.constraint_dict()
        for j in N:
            m.assign_1[j] = pmo.constraint(
                sum(x[i,j] for i in N) == 1)

        # relax everything for the bound solves,
        # since the objective uses a simple heuristic
        self.true_domain_type = pmo.ComponentMap()
        for xij in self.model.x.components():
            self.true_domain_type[xij] = xij.domain_type
            xij.domain_type = pmo.RealSet
        for yi in self.model.y.components():
            self.true_domain_type[yi] = yi.domain_type
            yi.domain_type = pmo.RealSet
        self._last_bound_was_feasible = False

        self.opt = pmo.SolverFactory(
            pyomo_solver,
            solver_io=pyomo_solver_io)

    def _check_feasible(self, tol=1e-4):
        """Check if the currently loaded solution
        is feasible for the true (binary) model."""
        for V in (self.model.x, self.model.y):
            for v in V.components():
                if (abs(v.value) > tol) and \
                   (abs(v.value - 1) > tol):
                    return False
        return True

    def _branch_y(self, parent):

        N = range(len(self.W))
        for i in N:
            yi = self.model.y[i]
            assert yi.lb in (0,1), \
                str(yi.name)+" "+str(yi.bounds)
            assert yi.ub in (0,1), \
                str(yi.name)+" "+str(yi.bounds)
            if yi.lb == 0:
                if yi.ub == 1:
                    break
                else:
                    assert yi.ub == 0
            else:
                assert yi.lb == 1
                assert yi.lb == yi.ub
        else:
            # there is no branching left to do
            return ()

        for k in range(i,len(self.W)):
            assert self.model.y[k].lb == 0
            assert self.model.y[k].ub in (0,1)

        orig_bounds = pmo.ComponentMap()
        orig_bounds.update((yi, yi.bounds)
                           for yi in self.model.y.components())
        orig_bounds.update((xij, xij.bounds)
                           for xij in self.model.x.components())

        children = [parent.new_child(),
                    parent.new_child()]

        # first branch: fix this bin on
        self.model.y[i].lb = self.model.y[i].ub = 1
        self.save_state(children[0])

        # second branch: fix this bin off, as well as all
        #                bins following it
        for k in range(i,len(self.W)):
            self.model.y[k].lb = self.model.y[k].ub = 0
            for j in N:
                assert self.model.x[k,j].lb == 0
                self.model.x[k,j].ub = 0
        self.save_state(children[1])

        # reset bounds
        for var in orig_bounds:
            var.bounds = orig_bounds[var]

        return children

    def _branch_x(self, parent):
        N = range(len(self.W))
        orig_bounds = pmo.ComponentMap()
        orig_bounds.update((yi, yi.bounds)
                           for yi in self.model.y.components())
        orig_bounds.update((xij, xij.bounds)
                           for xij in self.model.x.components())

        # do some simple preprocessing to reduce
        # the branch space
        for i in N:
            assert self.model.y[i].lb <= self.model.y[i].ub
            assert self.model.y[i].lb in (0,1)
            assert self.model.y[i].ub in (0,1)
            if self.model.y[i].ub == 0:
                for j in N:
                    assert self.model.x[i,j].lb == 0
                    self.model.x[i,j].ub = 0
        unassigned_items = []
        for j in N:
            assigned = sum(self.model.x[i,j].lb for i in N)
            if assigned == 0:
                unassigned_items.append(j)
            else:
                assert assigned == 1
                for i in N:
                    if self.model.x[i,j].lb == 0:
                        self.model.x[i,j].ub = 0
        # find an item that is not already fixed into a bin
        bv = None
        for j in unassigned_items:
            for i in N:
                if (self.model.x[i,j].lb == 0) and \
                   (self.model.x[i,j].ub == 1):
                    bv = self.model.x[i,j]
                    break
            if bv is not None:
                break
        else:
            return ()

        assert bv is not None
        children = [parent.new_child(),
                    parent.new_child()]
        bv.lb = bv.ub = 1
        self.save_state(children[0])
        bv.lb = bv.ub = 0
        self.save_state(children[1])

        # reset bounds
        for var in orig_bounds:
            var.bounds = orig_bounds[var]

        return children

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.minimize

    def objective(self):
        for xij in self.model.x.components():
            assert xij.lb in (0,1)
            assert xij.ub in (0,1)
            xij.value = xij.lb
        for yi in self.model.y.components():
            assert yi.lb in (0,1)
            assert yi.ub in (0,1)
            yi.value = yi.lb
        # a simple heuristic: iterate over items in random order
        #                     and fill up bins, adding as needed
        N = range(len(self.W))
        item_order = []
        for j in N:
            assigned = sum(bool(self.model.x[i,j].value == 1)
                           for i in N)
            if assigned == 1:
                for i in N:
                    if self.model.x[i,j].value == 1:
                        if self.model.y[i].ub == 0:
                            return self.infeasible_objective()
                        else:
                            self.model.y[i].value = 1
            else:
                assert assigned == 0
                item_order.append(j)

        on_bins = []
        off_bins = []
        for i in N:
            yi = self.model.y[i]
            if yi.value == 1:
                on_bins.append(i)
            else:
                assert yi.value == 0
                off_bins.append(i)
        bins = on_bins + off_bins
        bin_weight = {i: sum(self.W[j]*self.model.x[i,j].value for j in N)
                      for i in N}

        random.shuffle(item_order)
        for j in item_order:
            assert self.W[j] > 0
            for i in bins:
                if self.model.x[i,j].ub == 1:
                    if self.model.y[i].value == 0:
                        if self.model.y[i].ub == 1:
                            self.model.y[i].value = 1
                        else:
                            continue
                    if bin_weight[i] + self.W[j] <= self.V:
                        assert self.model.x[i,j].value == 0
                        assert self.model.x[i,j].lb == 0
                        assert self.model.x[i,j].ub == 1
                        self.model.x[i,j].value = 1
                        bin_weight[i] += self.W[j]
                        break
            else:
                return self.infeasible_objective()

        assert self._check_feasible()
        return self.model.objective()

    def bound(self):
        self._last_bound_was_feasible = False
        results = self.opt.solve(self.model, load_solutions=False)
        if (str(results.solver.status) == "ok") and \
           (str(results.solver.termination_condition) == "optimal"):
            self.model.load_solution(results.solution(0))
            self._last_bound_was_feasible = self._check_feasible()
            return round(self.model.objective(), 7)
        elif str(results.solver.termination_condition) == "unbounded":
            assert str(results.solver.status) in ("ok","warning")
            return self.unbounded_objective()
        elif str(results.solver.termination_condition) == "infeasible":
            assert str(results.solver.status) in ("ok","warning")
            return self.infeasible_objective()
        else:
            assert str(results.solver.status) == "error"
            assert "Restoration Phase Failed" in \
                results.solver.message
            # assume infeasible
            return self.infeasible_objective()

    def save_state(self, node):
        state_size = len(self.model.x)*2 + len(self.model.y)*2
        node.resize(state_size)
        state = node.state
        ndx = 0
        for xij in self.model.x.components():
            state[ndx] = xij.lb
            ndx += 1
        for xij in self.model.x.components():
            state[ndx] = xij.ub
            ndx += 1
        for yi in self.model.y.components():
            state[ndx] = yi.lb
            ndx += 1
        for yi in self.model.y.components():
            state[ndx] = yi.ub
            ndx += 1

    def load_state(self, node):
        state = node.state
        ndx = 0
        for xij in self.model.x.components():
            xij.lb = state[ndx]
            ndx += 1
        for xij in self.model.x.components():
            xij.ub = state[ndx]
            ndx += 1
        for yi in self.model.y.components():
            yi.lb = state[ndx]
            ndx += 1
        for yi in self.model.y.components():
            yi.ub = state[ndx]
            ndx += 1
        self._last_bound_was_feasible = False

    def branch(self, parent):
        if self._last_bound_was_feasible:
            return ()
        # try to branch on y
        children = self._branch_y(parent)
        if len(children):
            return children
        # otherwise, branch on x
        return self._branch_x(parent)

if __name__ == "__main__":
    import pybnb.misc

    V = 100
    W = [49,41,34,33,29,26,26,22,20,19]

    #V = 524
    #W = [442,252,252,252,252,252,252,252,
    #     127,127,127,127,127,
    #     106,106,106,106,
    #     85,84,46,37,37,12,12,12,
    #     10,10,10,10,10,10,9,9]

    problem = BinPacking(V,W)
    pybnb.misc.create_command_line_solver(problem)
