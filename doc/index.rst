Welcome to ``PyBnB``
====================

``pybnb`` is a parallel branch-and-bound engine written in
Python. It designed to run on distributed computing
architectures, using ``mpi4py`` and ``numpy`` for fast
inter-process communication.

This package is meant to serve as a back-end for
problem-specific solution strategies that utilize a
branch-and-bound algorithm. The following core functionality
is included:

 - work load distribution through a central dispatcher
 - work task prioritization strategies (e.g., worst bound
   first, breadth first, custom)
 - solver-like log output showing algorithm progress

To use this package, one must implement a branch-and-bound
problem by subclassing the :class:`Problem
<pybnb.problem.Problem>` interface and defining the methods
shown in the example below.

.. code-block:: pycon

    >>> import pybnb
    >>> # define a branch-and-bound problem
    >>> class MyProblem(pybnb.Problem):
    ...    def sense(self): ...
    ...    def objective(self): ...
    ...    def bound(self): ...
    ...    def save_state(self, node): ...
    ...    def load_state(self, node): ...
    ...    def branch(self, node): ...
    >>> # solve a problem instance
    >>> result = pybnb.solve(MyProblem())
    >>> print(result.solution_status)
    'optimal'

.. toctree::
    :maxdepth: 2

    getting_started/index.rst
    reference/index.rst
