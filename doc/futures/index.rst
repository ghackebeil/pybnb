pybnb.futures
=============

The `pybnb.futures` module stores utilities that are still
in the early phase of development. They will typically be
fairly well tested, but are subject to change or be removed
without much notice from one release to the next.

Using a Nested Solve to Improve Parallel Performance
----------------------------------------------------

The :class:`NestedSolver <pybnb.futures.NestedSolver>`
object is a wrapper class for problems that provides an easy
way to implement a custom two-layer, parallel
branch-and-bound solve. That is, a branch-and-bound solve
where, at the top layer, a single dispatcher serves nodes to
worker processes over MPI, and those workers process each
node by performing their own limited branch-and-bound solve
in serial, rather than simply evaluating the node bound and
objective and returning its immediate children to the
dispatcher.

The above strategy can be implemented by simply wrapping the
problem argument with this class before passing it to the
solver, as shown below.

.. code-block:: python

    results = solver.solve(
        pybnb.futures.NestedSolver(problem,
                                   queue_strategy=...,
                                   track_bound=...,
                                   time_limit=...,
                                   node_limit=...),
        queue_strategy='bound',
        ...)

The `queue_strategy`, `track_bound`, `time_limit`, and
`node_limit` solve options can be passed into the
:class:`NestedSolver <pybnb.futures.NestedSolver>` class
when it is created to control these aspects of the
sub-solves used by the workers when processing a node.

This kind of scheme can be useful for problems with
relatively fast bound and objective computations, where the
overhead of updates to the central dispatcher over MPI is a
clear bottleneck. It is important to consider, however, that
assigning large values to the `node_limit` or `time_limit`
nested solve options may result in more work being performed
to achieve the same result as the non-nested case. As such,
the use of this solution scheme may not always result in a
net benefit for the total solve time.

Next, we show how this class is used to maximize the
parallel performance of the `TSP example
<https://github.com/ghackebeil/pybnb/blob/master/examples/scripts/tsp/tsp_byvertex.py>`_.
Tests are run using CPython 3.7 and PyPy3 6.0 (Python 3.5.3)
on a laptop with a single quad-core 2.6 GHz Intel Core i7
processor.

The code block below shows the main call to the solver used
in the TSP example, except it has been modified so that the
original problem is passed to the solver (no nested solve):

.. code-block:: python
  :emphasize-lines: 2

    results = solver.solve(
        problem,
        queue_strategy='depth',
        initialize_queue=queue,
        best_node=best_node,
        objective_stop=objective_stop)

Running the serial case as follows,

.. code-block:: console

    $ python -O tsp_naive.py fri26_d.txt

on CPython 3.7 we achieve a peak performance of ~19k nodes
processed per second, and on PyPy3 6.0 the performance peaks
at ~150k nodes processed per second. Compare this with the
parallel case (using three workers and one dispatcher),

.. code-block:: console

    $ mpirun -np 4 python -O tsp_naive.py fri26_d.txt

where with CPython 3.7 we achieve a peak performance of ~21k
nodes per second, and with PyPy3 6.0 the performance
actually drops to ~28k nodes per second (nowhere near the 3x
increase one would hope for).

Now consider the TSP example in its original form, where the
problem argument is wrapped with the :class:`NestedSolver
<pybnb.futures.NestedSolver>` object:

.. code-block:: python
  :emphasize-lines: 2,3,4,5

    results = solver.solve(
        pybnb.futures.NestedSolver(problem,
                                   queue_strategy='depth',
                                   track_bound=False,
                                   time_limit=1),
        queue_strategy='depth',
        initialize_queue=queue,
        best_node=best_node,
        objective_stop=objective_stop)

Running the parallel case, with CPython 3.7 we achieve a
peak performance of ~60k nodes per second, and with PyPy3
6.0 we achieve ~450k nodes per second!
