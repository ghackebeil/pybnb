Welcome to ``PyBnB``
====================

``pybnb`` is a parallel branch-and-bound engine designed to
run on distributed-memory computing architectures. It uses
the ``mpi4py`` Python package for message passing.

.. code-block:: pycon

    >>> import pybnb
    >>> # define a branch-and-bound problem
    >>> class MyProblem(pybnb.Problem):
    ...    def sense(self): ...
    ...    def objective(self): ...
    ...    def bound(self): ...
    ...    def save_state(self, node): ...
    ...    def load_state(self, node): ...
    ...    def branch(self, parent_node): ...
    >>> # solve a problem instance
    >>> result = pybnb.solve(MyProblem())
    >>> print(result.solution_status)
    'optimal'

Getting Started
===============

Under construction...

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    reference/common
    reference/problem
    reference/node
    reference/solver
    reference/pyomo_tools
