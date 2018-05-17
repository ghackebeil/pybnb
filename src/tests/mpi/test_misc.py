import os
import tempfile
import logging

import pytest
from runtests.mpi import MPITest

from pybnb.common import minimize
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.node import Node
from pybnb.solver import Solver
from pybnb.problem import Problem
from pybnb.dispatcher import DispatcherQueueData
from pybnb.misc import get_simple_logger

from six import StringIO

from .common import mpi_available


def _get_logging_baseline(size):
    out = \
"""[DEBUG] 0: debug
[INFO] 0: info
[WARNING] 0: warning
[ERROR] 0: error"""
    for i in range(1,size):
        out += ("""
[DEBUG] %d: debug
[INFO] %d: info
[WARNING] %d: warning
[ERROR] %d: error""") % (i,i,i,i)
    return out

class DummyProblem(Problem):

    def sense(self): return minimize
    def objective(self): return 0.0
    def bound(self): return 0.0
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self, parent): return ()

def _logging_check(comm):
    opt = Solver(comm=comm)
    p = DummyProblem()
    if opt.is_dispatcher:
        assert (comm is None) or (comm.rank == 0)
        root = Node()
        p.save_state(root)
        root.bound = p.unbounded_objective
        assert root.tree_id is None
        Node._insert_tree_id(root._data, 0)
        initialize_queue = DispatcherQueueData(
            nodes=[root],
            next_tree_id=1)
        out = StringIO()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        opt._disp.initialize(p.infeasible_objective,
                             initialize_queue,
                             "bound",
                             ConvergenceChecker(p.sense()),
                             None, None,
                             get_simple_logger(console=True,
                                               stream=out,
                                               level=logging.DEBUG,
                                               formatter=formatter),
                             0.0)
        opt._disp.log_debug("0: debug")
        opt._disp.log_info("0: info")
        opt._disp.log_warning("0: warning")
        opt._disp.log_error("0: error")
        if (comm is not None) and (comm.size > 1):
            opt._disp.serve()
    else:
        assert comm is not None
        if comm.size > 1:
            for i in range(1, comm.size):
                if comm.rank == i:
                    opt._disp.log_debug(str(comm.rank)+": debug")
                    opt._disp.log_info(str(comm.rank)+": info")
                    opt._disp.log_warning(str(comm.rank)+": warning")
                    opt._disp.log_error(str(comm.rank)+": error")
                opt.worker_comm.Barrier()
            if opt.worker_comm.rank == 0:
                opt._disp.stop_listen()
    if comm is not None:
        comm.Barrier()
    if opt.is_dispatcher:
        assert ('\n'.join(out.getvalue().splitlines()[7:])) == \
                _get_logging_baseline(comm.size if comm is not None else 1)

def test_logging_nocomm():
    _logging_check(None)

if mpi_available:

    @MPITest(commsize=[1, 2, 3])
    def test_bad_dispatcher_rank(comm):
        with pytest.raises(ValueError):
            Solver(comm=comm, dispatcher_rank=-1)
        with pytest.raises(ValueError):
            Solver(comm=comm, dispatcher_rank=comm.size)
        with pytest.raises(ValueError):
            Solver(comm=comm, dispatcher_rank=comm.size - 1.1)
        Solver(comm=comm, dispatcher_rank=comm.size - 1)

    @MPITest(commsize=[1, 2, 3])
    def test_logging(comm):
        _logging_check(comm)
