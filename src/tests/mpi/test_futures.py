import logging

from pybnb.common import minimize
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.node import Node
from pybnb.solver import Solver
from pybnb.problem import Problem
from pybnb.dispatcher import DispatcherQueueData
from pybnb.misc import get_simple_logger
from pybnb.futures import _RedirectHandler

from .common import mpi_available

from six import StringIO
from runtests.mpi import MPITest

def _get_logging_baseline(size):
    out = \
"""[WARNING] 0: warning
[ERROR] 0: error
[CRITICAL] 0: critical"""
    for i in range(1,size):
        out += ("""
[WARNING] %d: warning
[ERROR] %d: error
[CRITICAL] %d: critical""") % (i,i,i)
    return out

class DummyProblem(Problem):

    def sense(self): return minimize
    def objective(self): return 0.0
    def bound(self): return 0.0
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self): return ()

def _logging_redirect_check(comm):
    opt = Solver(comm=comm)
    p = DummyProblem()
    log = logging.Logger(None,
                         level=logging.WARNING)
    log.addHandler(_RedirectHandler(opt._disp))
    if opt.is_dispatcher:
        assert (comm is None) or (comm.rank == 0)
        root = Node()
        p.save_state(root)
        root.objective = p.infeasible_objective()
        root.bound = p.unbounded_objective()
        initialize_queue = DispatcherQueueData(
            nodes=[root],
            worst_terminal_bound=None,
            sense=p.sense())
        out = StringIO()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        opt._disp.initialize(p.infeasible_objective(),
                             None,
                             initialize_queue,
                             "bound",
                             ConvergenceChecker(p.sense()),
                             None, None, None, True,
                             get_simple_logger(console=True,
                                               stream=out,
                                               level=logging.DEBUG,
                                               formatter=formatter),
                             0.0,
                             True)
        log.debug("0: debug")
        log.info("0: info")
        log.warning("0: warning")
        log.error("0: error")
        log.critical("0: critical")
        if (comm is not None) and (comm.size > 1):
            opt._disp.serve()
    else:
        assert comm is not None
        if comm.size > 1:
            for i in range(1, comm.size):
                if comm.rank == i:
                    log.debug(str(comm.rank)+": debug")
                    log.info(str(comm.rank)+": info")
                    log.warning(str(comm.rank)+": warning")
                    log.error(str(comm.rank)+": error")
                    log.critical(str(comm.rank)+": critical")
                opt.worker_comm.Barrier()
            if opt.worker_comm.rank == 0:
                opt._disp.stop_listen()
    if comm is not None:
        comm.Barrier()
    if opt.is_dispatcher:
        assert ('\n'.join(out.getvalue().splitlines()[7:])) == \
                _get_logging_baseline(comm.size if comm is not None else 1)

def test_logging_redirect_nocomm():
    _logging_redirect_check(None)

if mpi_available:

    @MPITest(commsize=[1, 2, 3])
    def test_logging_redirect(comm):
        _logging_redirect_check(comm)
