Advanced Usage
==============

Setting the Queue Strategy and Solver Tolerances
------------------------------------------------

``pybnb`` uses a default queue strategy that prioritizes
improving the global optimality bound over other solve
metrics. The `queue_strategy` solve option controls this
behavior. See the :class:`QueueStrategy
<pybnb.common.QueueStrategy>` enum for a complete list of
available strategies.

The best queue strategy to use depends on characteristics of
the problem being solved. Queue strategies such as "depth"
and "lifo" tend to keep the queue size small and reduce the
dispatcher overhead, which may be important for problems
with relatively fast objective and bound
evaluations. Setting the `track_bound` solve option to false
will further reduce the dispatcher overhead of these queue
strategies. On the other hand, using these strategies may
result in a larger number of nodes being processed before
reaching a given optimality gap.

The `absolute_gap` and `relative_gap` solve options can be
adjusted to control when the solver considers a solution to
be optimal. By default, optimality is defined as having an
absolute gap of zero between the best objective and the
global problem bound, and no relative gap is considered.
(`absolute_gap=0`, `relative_gap=None`). To enable a check
for relative optimality, simply assign a non-negative value
to the `relative_gap` solver option (e.g.,
`relative_gap=1e-4`). Additionally, a function can be
provided through the `scale_function` solver option for
computing the scaling factor used to convert an absolute gap
to a relative gap. This function should have the signature
`f(bound, objective) -> float`. The default scale function
is `max{1.0,|objective|}`.

Two additional solve options to be aware of are the
`queue_tolerance` and `branch_tolerance`.  The
`queue_tolerance` setting controls when new child nodes are
allowed into the queue. If left unset, it will be assigned
the value of the `absolute_gap` setting. It is not affected
by the `relative_gap` setting. See the section titled
:ref:`continuing` for further discussion along with an
example. Finally, the `branch_tolerance` setting controls
when the `branch` method is called. The default setting of
zero means that any non-zero gap between a node's local
bound and objective will allow branching. Larger settings
may be useful for avoiding tolerance issues in a problem
implementation.

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

.. _continuing:

Continuing a Solve After Stopping
---------------------------------

It is possible to continue a solve with new termination
criteria, starting with the candidate solution and remaining queued
nodes from a previous solve. The following code block shows how
this can be done.

.. code-block:: python

    solver = pybnb.Solver()
    results = solver.solve(problem,
                           absolute_gap=1e-4,
                           queue_tolerance=1e-8,
                           time_limit=10)
    queue = solver.save_dispatcher_queue()
    results = solver.solve(problem,
                           best_objective=results.objective,
                           best_node=results.best_node,
                           initialize_queue=queue,
                           absolute_gap=1e-8)

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

Note the use of the `queue_tolerance` solve option in the
first solve above. If left unused, this option will be set
equal to the value of the `absolute_gap` setting (it is not
affected by the `relative_gap` setting). The
`queue_tolerance` setting determines when new child nodes
are eligible to enter the queue. If the difference between a
child node's bound estimate and the best objective is less
than or equal to the `queue_tolerance` (or worse than the
best objective by any amount), the child node will be
discarded. Thus, in the example above, the first solve uses
a `queue_tolerance` equal to the `absolute_gap` used in the
second solve to avoid discarding child nodes in the first
solve that may be required to achieve the tighter optimality
settings used in the second solve.

Assigning the :attr:`objective
<pybnb.solver_results.SolverResults.objective>` attribute of
the results object to the `best_objective` solve option is
only necessary if (1) the initial solve was given a
`best_objective` and the solver did not obtain a best node
with a matching objective, or (2) if the initial solve is
unbounded.  In the latter case, the :attr:`best_node
<pybnb.solver_results.SolverResults.best_node>` attribute of
the results object will be None and the dispatcher queue
will be empty, so the unboundedness of the problem can only
be communicated to the next solve via the `best_objective`
solve option.  If one is careful about checking the status
of the solution and no initial best objective is used (both
recommended), then the `best_objective` solver option can be
left unused, as shown below:

.. code-block:: python

    solver = pybnb.Solver()
    results = solver.solve(problem,
                           asolute_gap=1e-4,
                           queue_tolerance=1e-8,
                           time_limit=10)
    if results.solution_status in ("optimal",
                                   "feasible"):
        queue = solver.save_dispatcher_queue()
        results = solver.solve(problem,
                               best_node=results.best_node,
                               initialize_queue=queue,
                               absolute_gap=1e-8)

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
COMPRESSION                 bool    False                   | Indicates if serialized node state should be
                                                            | compressed using zlib.
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
