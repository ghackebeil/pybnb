from pybnb.solver import Solver

from .common import mpi_available

from runtests.mpi import MPITest

def test_solver_nocomm():
    solver = Solver(comm=None)
    assert solver.is_worker
    assert solver.is_dispatcher
    assert solver.comm is None
    assert solver.worker_comm is None

if mpi_available:


    @MPITest(commsize=[1, 2, 3])
    def test_solver_comm(comm):
        solver = Solver(comm=comm, dispatcher_rank=0)
        if comm.size > 1:
            if comm.rank == 0:
                assert solver.is_dispatcher
                assert not solver.is_worker
                assert solver.comm is comm
                assert solver.worker_comm is None
            else:
                assert not solver.is_dispatcher
                assert solver.is_worker
                assert solver.comm is comm
                assert solver.worker_comm is not None
        else:
            assert solver.is_dispatcher
            assert solver.is_worker
            assert solver.comm is comm
            assert solver.worker_comm is comm
