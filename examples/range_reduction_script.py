import pybnb
from pybnb.pyomo_tools import RangeReductionProblem

from rosenbrock_2d import Rosenbrock2D

class Rosenbrock2D_RangeReduction(RangeReductionProblem):

    improved_abstol = 1e-3

    #
    # Implement RangeReductionProblem abstract methods
    #

    def range_reduction_model_setup(self):
        """Called prior to starting range reduction solves
        to set up the Pyomo model"""
        self.problem.setup_model_for_bound()

    def range_reduction_objective_changed(self, objective):
        """Called to notify that the range reduction routine
        has changed the objective"""
        # nothing to do for this problem
        pass

    def range_reduction_constraint_added(self, constraint):
        """Called to notify that the range reduction routine
        has added a constraint"""
        # nothing to do for this problem
        pass

    def range_reduction_constraint_removed(self, constraint):
        """Called to notify that the range reduction routine
        has removed a constraint"""
        # nothing to do for this problem
        pass

    def range_reduction_get_objects(self):
        """Called to collect the set of objects over which
        to perform range reduction solves"""
        vlist = []
        x, y = self.problem._model.x, self.problem._model.y
        if (y.ub - y.lb) > \
           self.problem._branch_abstol:
            vlist.append(y)
        if (x.ub - x.lb) > \
           self.problem._branch_abstol:
            vlist.append(x)
        return vlist

    def range_reduction_solve_for_object_bound(self, x):
        """Called to perform a range reduction solve for a
        Pyomo model object"""
        results = self.problem._ipopt.solve(self.problem._model,
                                            load_solutions=False)
        if str(results.solver.termination_condition) == "optimal":
            assert str(results.solver.status) == "ok"
            symbol_map = results.solution(0).symbol_map
            assert results.solution(0).default_variable_value is None
            return results.solution(0).\
                variable[symbol_map.getSymbol(x)]['Value']

    def range_reduction_model_cleanup(self):
        """Called after range reduction has finished to
        allow the user to execute any cleanup to the Pyomo
        model."""
        # nothing to do for this problem
        pass

    def range_reduction_process_bounds(self,
                                       objects,
                                       lower_bounds,
                                       upper_bounds):
        """Called to process the bounds obtained by the
        range reduction solves"""
        changed_cnt = 0
        for lb, ub, var in zip(lower_bounds,
                               upper_bounds,
                               objects):
            changed = False
            if lb > var.lb + self.improved_abstol:
                var.lb = lb
                changed = True
            if ub < var.ub - self.improved_abstol:
                var.ub = ub
                changed = True
            if changed:
                changed_cnt += 1
        self.problem.rebuild_convex_envelopes()
        return False

if __name__ == "__main__":
    import argparse

    comm = None
    try:
        import mpi4py.MPI
        comm = mpi4py.MPI.COMM_WORLD
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description=("Run parallel branch and bound "
                     "with optimality-based range reduction "
                     "on the first few nodes"))
    parser.add_argument("--results-filename", type=str, default=None,
                        help=("When set, saves the solver results into a "
                              "YAML-formated file with the given name."))
    args = parser.parse_args()

    problem = Rosenbrock2D(xL=-25, xU=25,
                           yL=-25 ,yU=25)

    best_objective = None
    if (comm is None) or (comm.rank == 0):
        best_objective = problem.objective()
    if (comm is not None):
        best_objective = comm.bcast(best_objective, root=0)
    assert best_objective != problem.unbounded_objective

    # do parallel bounds tightening for the first three nodes
    obrr = Rosenbrock2D_RangeReduction(
        problem,
        best_objective,
        comm=comm)

    dispatcher_queue = None
    best_objective = None
    if (comm is None) or (comm.rank == 0):
        opt_obrr = pybnb.Solver(comm=None)
        results = opt_obrr.solve(obrr, node_limit=7,
                                 best_objective=best_objective)
        dispatcher_queue = opt_obrr.save_dispatcher_queue()
        best_objective = results.objective
    else:
        obrr.listen(root=0)
    del obrr

    # continue the solve without bounds tightening at the
    # remaining nodes
    results = pybnb.solve(problem,
                          comm=comm,
                          dispatcher_rank=0,
                          best_objective=best_objective,
                          initialize_queue=dispatcher_queue,
                          results_filename=args.results_filename)
