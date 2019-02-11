Advanced Usage
==============

Terminating a Solve Early
-------------------------

A solve that is launched without the use of `mpiexec` can be
terminated at any point by entering `Ctrl-C` (sending the
process a `SIGINT` signal). If the signal is successfully
received, the solver will attempt to gracefully stop the
solve after it finishes processing the current node, and it
will mark the :attr:`termination_condition
<pybnb.solver_results.SolverResults.termination_condition>`
attribute of the solver results object with the
:attr:`interrupted
<pybnb.common.TerminationCondition.interrupted>` status.

Solves launched through `mpiexec` typically can not be
gracefully terminated using the `Ctrl-C` method. This is due
to the way the MPI process manager handles the `SIGINT`
signal. However, the solve can be gracefully terminated by
sending a `SIGUSR1` signal to the dispatcher process (this
also works for the case when the solve was launched without
`mpiexec`). The pid and hostname of the dispatcher process
are always output at the beginning of the solve.

.. code-block:: console

    $ mpiexec -n 4 python simple.py
    Starting branch & bound solve:
     - dispatcher pid: <pid> (<hostname>)
    ...

Assuming one is logged in to the host where the dispatcher
process is running, the solve can be terminated using a
command such as:

.. code-block:: console

    $ kill -USR1 <pid>

Continuing a Solve After Stopping
---------------------------------

It is possible to continue a solve with new termination
criteria, starting with the candidate solution and remaining queued
nodes from a previous solve. The following code block shows how
this can be done.

.. code-block:: python

    solver = pybnb.Solver()
    results = solver.solve(problem,
                           node_limit=10)
    queue = solver.save_dispatcher_queue()
    solver.solve(problem,
                 best_node=results.best_node,
                 initialize_queue=queue)

For the dispatcher process, the :func:`save_dispatcher_queue
<pybnb.solver.Solver.save_dispatcher_queue>` method returns
an object of type :class:`DispatcherQueueData
<pybnb.dispatcher.DispatcherQueueData>`, which can be
assigned to the `initialize_queue` keyword of the
:func:`solve <pybnb.solver.Solver.solve>` method. For
processes that are not the dispatcher, this function returns
`None`, which is the default value of the `initialize_queue`
keyword. The :attr:`best_node
<pybnb.solver_results.SolverResults.best_node>` attribute of
the results object will be identical for all processes
(possibly equal to None), and can be directly assigned to
the `best_node` solver option.

.. _configuration:

Serialization Configuration
---------------------------

The following configuration items are available for
controlling how node state is transmitted during a parallel
solve:

=========================== ======= ======================= =======
config item                 type    default                 meaning
=========================== ======= ======================= =======
SERIALIZER                  str     "pickle"                | The serializer used to transform the user-defined
                                                            | node state into a byte stream that can be
                                                            | transmitted with MPI. Allowed values are "pickle"
                                                            | and "dill".
SERIALIZER_PROTOCOL_VERSION int     pickle.HIGHEST_PROTOCOL | The value assigned to the ``protocol`` keyword of
                                                            | the pickle or dill ``dumps`` function.
MARSHAL_PROTOCOL_VERSION    int     2                       | The value assigned to the ``version`` argument of
                                                            | the ``marshal.dumps`` function. The marshal module
                                                            | is used to serialize all other node attributes
                                                            | besides the user-defined state. It is unlikely
                                                            | that this setting would need to be adjusted.
=========================== ======= ======================= =======

These settings are available as attributes on the
``pybnb.config`` object. This object can be modified by the
user to, for instance, change the serializer for the
user-defined node state to the ``dill`` module. To do so,
one would add the following to the beginning of their code::

  pybnb.config.SERIALIZER = "dill"

Each of these settings can also be modified through the
environment by exporting a variable with ``PYBNB_``
prepended to the attribute name on the config object::

  export PYBNB_SERIALIZER=pickle

The environment is checked during the first import of
``pybnb``, so when configurations are applied by directly
modifying the ``pybnb.config`` object, this will override
those applied through environment variables.  The
:func:`pybnb.config.reset(...)
<pybnb.configuration.Configuration.reset>` method can be
called to restore all configuration options to their default
setting (ignoring the environment if specified).

pybnb.futures
-------------
The `pybnb.futures` module stores utilities that are still
in the early phase of development. They will typically be
fairly well tested, but are subject to change or be removed
without much notice from one release to the next.

Using a Nested Solve to Improve Parallel Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :class:`NestedSolver <pybnb.futures.NestedSolver>`
object is a wrapper class for problems that provides an easy
way to implement a custom two-layer, parallel branch-and-bound
solve. That is, a branch-and-bound solve where,
at the top layer, a single dispatcher serves nodes to worker
processes over MPI, and those workers process each node by
performing their own limited branch-and-bound solve in
serial rather than simply evaluating the node bound and
objective and returning its immediate children.

The above strategy can be implemented by simply wrapping the
problem argument with this class before passing it to the
solver, as shown below.

.. code-block:: python

    results = solver.solve(
        pybnb.futures.NestedSolver(problem,
                                   queue_strategy=...,
                                   time_limit=...,
                                   node_limit=...),
        queue_strategy='bound',
        ...)

The `queue_strategy`, `time_limit`, and `node_limit` solve
options can be passed into the :class:`NestedSolver
<pybnb.futures.NestedSolver>` class when it is created to
control these aspects of the sub-solves used by the workers
when processing a node.

This kind of scheme can be useful for problems with
relatively fast bound and objective computations, where the
overhead of updates to the central dispatcher over MPI is a
clear bottleneck. Next, we show how this class is used to
maximize the parallel performance of the `TSP example
<https://github.com/ghackebeil/pybnb/blob/master/examples/scripts/tsp/tsp_naive.py>`_.
Tests are run using CPython 3.7 and PyPy3 6.0 (Python 3.5.3) on a laptop with
a single quad-core 2.6 GHz Intel Core i7 processor.

The code block below shows the main call to the solver used
in the TSP example, except it has been modified so that the
original problem is passed to the solver (no nested solve):

.. code-block:: python
  :emphasize-lines: 2

    results = solver.solve(
        problem,
        absolute_gap=0,
        relative_gap=None,
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
  :emphasize-lines: 2,3,4

    results = solver.solve(
        pybnb.futures.NestedSolver(problem,
                                   queue_strategy='depth',
                                   time_limit=1),
        absolute_gap=0,
        relative_gap=None,
        queue_strategy='depth',
        initialize_queue=queue,
        best_node=best_node,
        objective_stop=objective_stop)

Running the parallel case, with CPython 3.7 we achieve a
peak performance of ~60k nodes per second, and with PyPy3
6.0 we achieve ~450k nodes per second!
