Getting Started
===============

Installation
------------
You can install ``pybnb`` with ``pip``:

.. code-block:: console

    $ pip install pybnb

``pybnb`` requires ``mpi4py`` to solve problems in
parallel. However, it will also solve problems in serial if
this module is not available. Thus, ``mpi4py`` is not listed
as a package requirement, and it may need to be installed in
a separate step.

Complete Example
----------------

The code below shows a complete example script that (1) defines
a problem, (2) creates a solver, and (3) solves the problem.

.. literalinclude:: ../../examples/scripts/simple.py
   :language: python
   :prepend: # simple.py
   :lines: 16-

To solve the problem in serial, the example script should be
launched with the python interpretor:

.. code-block:: console

    $ python simple.py

To solve the problem in parallel, the example script should
be launched using the same command as above, only wrapped
with ``mpiexec`` (specifying the number processes):

.. code-block:: console

    $ mpiexec -n 4 python simple.py

Note that the parallel solve implementation used by
``pybnb`` always designates exactly one process as a
dispatcher.  If more than one process is involved in a
solve, the dispatcher will only manage the global work
queue, leaving the processing of all branch-and-bound nodes
to the remaining processes. Thus, one should not expect any
parallel speedup until at least three processes are used to
solve a problem.

Defining a Problem
------------------
To define a branch-and-bound problem with ``pybnb``, one
must define a class that implements the :class:`Problem
<pybnb.problem.Problem>` interface, which includes defining
at least the six required methods shown below.

.. code-block:: python

    import pybnb
    class MyProblem(pybnb.Problem):
       def __init__(self): ...
       # required methods
       def sense(self): ...
       def objective(self): ...
       def bound(self): ...
       def save_state(self, node): ...
       def load_state(self, node): ...
       def branch(self, node): ...
       # optional methods
       def notify_solve_begins(self,
                               comm,
                               worker_comm,
                               convergence_checker):
           ...
       def notify_new_best_objective_received(self,
                                              objective):
           ...
       def notify_new_best_objective(self,
                                     objective):
           ...
       def notify_solve_finished(self,
                                 comm,
                                 worker_comm,
                                 results):
           ...

.. note::
    The :class:`Problem <pybnb.problem.Problem>` base
    class is a purely abstract interface that adds no
    additional data to a problem implementation. It is not
    required to call ``Problem.__init__`` when defining the
    ``__init__`` method on a derived class.

The remainder of this section includes a detailed
description of each of the required methods.

 - :func:`Problem.sense() <pybnb.problem.Problem.sense>`

   This is the easiest method to define for a
   branch-and-bound problem. It should return the objective
   sense of the problem, which should always be one of
   :obj:`minimize <pybnb.common.minimize>` or
   :obj:`maximize <pybnb.common.maximize>`, and should
   not change what it returns over the lifetime of a
   problem. For instance, to define a problem with an
   objective value that should be minimized, the
   implementation would look something like:

   .. code-block:: python

       class MyProblem(pybnb.Problem):
          def sense(self):
              return pybnb.minimize

   The :class:`Problem <pybnb.problem.Problem>` base
   class defines two additional convenience methods
   :func:`Problem.infeasible_objective()
   <pybnb.problem.Problem.infeasible_objective>` and
   :func:`Problem.unbounded_objective()
   <pybnb.problem.Problem.unbounded_objective>` that return
   `+inf` or `-inf`, depending on the return value of
   :func:`Problem.sense() <pybnb.problem.Problem.sense>`.

 - :func:`Problem.bound() <pybnb.problem.Problem.bound>`

   This method should return a valid bound for the objective
   function over the current problem domain (as defined by
   the current problem state), or it can return
   :func:`self.ubnounded_objective()
   <pybnb.problem.Problem.unbounded_objective>` if a finite
   bound can not be determined.

 - :func:`Problem.objective() <pybnb.problem.Problem.objective>`

   This method should return a value for the objective
   function that is feasible for the current problem domain
   (as defined by the current problem state), or it can
   return :func:`self.infeasible_objective()
   <pybnb.problem.Problem.infeasible_objective>` if a
   feasible objective value can not be determined.

 - :func:`Problem.save_state(node) <pybnb.problem.Problem.save_state>`

   This method should save any relevant state information
   about the problem into the numeric :attr:`Node.state
   <pybnb.node.Node.state>` array on the node argument. The
   amount of available storage in this array can be adjusted
   by calling the :func:`resize <pybnb.node.Node.resize>`
   method on the node.

 - :func:`Problem.load_state(node) <pybnb.problem.Problem.load_state>`

   This method should load the problem state stored inside
   the numeric :attr:`Node.state <pybnb.node.Node.state>` array
   on the node argument. For instance, a pair of
   :func:`Problem.save_state <pybnb.problem.Problem.save_state>` and
   :func:`Problem.load_state <pybnb.problem.Problem.load_state>`
   implementations might look like:

   .. code-block:: python

       class MyProblem(pybnb.Problem):
           def __init__(self):
               self._xL = 0.0
               self._xU = 1.0
           def save_state(self, node):
               node.resize(2)
               node.state[0] = self._xL
               node.state[1] = self._xU
           def load_state(self, node):
               assert len(node.state) == 2
               self._xL = float(node.state[0])
               self._xU = float(node.state[1])

 - :func:`Problem.branch(node) <pybnb.problem.Problem.branch>`

   This method should partition the problem domain defined
   within the user state on the `node` object into
   zero or more child states and return them as new node
   objects. A child node should be created by calling
   :func:`node.new_child()
   <pybnb.node.Node.new_child>`. Note that for the branching
   process to make sense in the context of a
   branch-and-bound solve, the bound computed from the child
   node states should improve (or not be worse than) the
   bound for the parent node. Once the child bound is
   computed, if it is found to be worse than the bound from
   the parent node then the branch-and-bound solver will
   issue a warning, as this is likely indicative of a
   programming error or other numerical issues.

   When this method is called, the :attr:`node.bound
   <pybnb.node.Node.bound>` and :attr:`node.objective
   <pybnb.node.Node.objective>` attributes will have been
   set to the value returned from :func:`Problem.bound()
   <pybnb.problem.Problem.bound>` and
   :func:`Problem.objective()
   <pybnb.problem.Problem.objective>`, respectively.  Any
   child nodes returned from :func:`node.new_child()
   <pybnb.node.Node.new_child>` will inherit this bound and
   objective, which may affect their prioritization in the
   global work queue. As user can assign a new value to one
   or both of these attributes before returning a child
   node.

How the Solver Calls the Problem Methods
----------------------------------------

The following block of pseudocode provides a high level
overview of how the solver calls the methods on a
user-defined problem. Highlighted lines show where problem
methods are called.

.. code-block:: python
  :emphasize-lines: 5,6,8,16,17,18,20,23,25,30,31
  :linenos:

    def solve(problem, ...):
        #
        # solve initialization
        #
        sense = problem.sense()
        problem.notify_solve_begins(...)
        root = Node()
        problem.save_state(root)

        #
        # solve loop
        #
        while <solve_not_terminated>:
            node, best_objective = dispatcher.update(...)
            if <conditional_1>:
               problem.notify_new_best_objective_received(...)
            problem.load_state(node)
            bound = problem.bound()
            if <conditional_2>:
                objective = problem.objective()
                if <conditional_3>:
                    best_objective = objective
                    problem.notify_new_best_objective(...)
                if <conditional_4>:
                    problem.branch(node)

        #
        # solve finalization
        #
        problem.load_state(root)
        problem.notify_solve_finished(...)

Note from the above that, during the main solve loop
starting on line 13, it is safe to assume that the six
highlighted problem methods between line 13 and line 25 will
be called in the relative order shown. The conditions under
a subset of the methods will be called are briefly discussed
below:

 - **<conditional_1>** (line 15): This condition is met when
   the `best_objective` received from the dispatcher is not
   unbounded and improves upon the best objective currently
   known to the worker process. By default, the check for
   improvement is exact, but it can be relaxed by assigning
   a nonzero value to the `comparison_tolerance` keyword of
   the :func:`Solver.solve <pybnb.solver.Solver.solve>`
   method.
 - **<conditional_2>** (line 19): This condition is met when the
   bound computed by the problem for the current node is
   eligible for the queue relative to `best_objective` known
   to the process. By default, this is true when the bound
   is better than the `best_objective` by any nonzero
   amount, but this can be influenced by changing the
   default value of the `queue_tolerance` keyword of the
   :func:`Solver.solve <pybnb.solver.Solver.solve>` method.
 - **<conditional_3>** (line 21): This condition is met when
   the objective computed by the problem for the current
   node is not unbounded and improves upon the
   `best_objective` currently known to the process. By
   default, the check for improvement is exact, but it can
   be relaxed by assigning a nonzero value to the
   `comparison_tolerance` keyword of the :func:`Solver.solve
   <pybnb.solver.Solver.solve>` method.
 - **<conditional_4>** (line 24): This condition is met when
   the objective computed by the problem for the current
   node is not unbounded, when **<conditional_2>** is still
   satisfied, and when the difference between the node's
   bound and objective satisfies the branching tolerance. By
   default, the branching tolerance is zero, meaning that
   any distance between these two values will satisfy this
   check, but this can be adjusted using the
   `branching_tolerance` keyword of the :func:`Solver.solve
   <pybnb.solver.Solver.solve>` method.

Solving a Problem
-----------------
There are two approaches to solving a branch-and-bound
problem with ``pybnb``. The first is to simply call the
:func:`solve <pybnb.solver.solve>` convenience function,
and the second is to create a :class:`Solver
<pybnb.solver.Solver>` object directly and call the
:func:`Solver.solve <pybnb.solver.Solver.solve>` method.

Both approaches can solve a problem in serial or
parallel. The difference between the two approaches is that
the :func:`solve <pybnb.solver.solve>` convenience function
automatically creates a :class:`Solver <pybnb.solver.Solver>`
object and provides a few additional keywords that simplify
the process of saving solver output and results to a
file. Additionally, this function collects and reports
workload and timing statistics for the solve, which can add
some overhead. These statistics are not collected by the
:func:`Solver.solve <pybnb.solver.Solver.solve>` method,
thus, the second approach of directly using a :class:`Solver
<pybnb.solver.Solver>` can be more efficient.

Creating a Solver
^^^^^^^^^^^^^^^^^

The following example shows how to create a solver object.

.. code-block:: python

    import pybnb
    solver = pybnb.Solver()

By default, the solver will automatically use
``mpi4py.MPI.COMM_WORLD`` as the communicator, and the rank
0 process will act as the dispatcher. If the ``mpi4py``
module is not available, this will result in an
``ImportError``. The optional keywords ``comm`` and
``dispatcher_rank`` can be used to change the default
behavior.

When a solver is created with ``Solver(comm=None)``, this
will disable any attempted import of ``mpi4py``, allowing
problems to be solved without the use of any parallel
functionality. The ``comm`` keyword can also be assigned a
communicator different from ``mpi4py.MPI.COMM_WORLD``. If
the solver communicator includes more than one process, the
``dispatcher_rank`` keyword can be assigned a process rank
to control which process is designated as the dispatcher.
However the solver is initialized, the following assertions
hold true for the :attr:`is_dispatcher
<pybnb.solver.Solver.is_dispatcher>` and :attr:`is_worker
<pybnb.solver.Solver.is_worker>` attributes of the solver
object.

.. code-block:: python

    if (solver.comm is None) or \
       (solver.comm.size == 1):
        assert solver.is_dispatcher and \
            solver.is_worker
    else:
        if solver.comm.rank == <dispatcher_rank>:
            assert solver.is_dispatcher and \
                (not solver.is_worker)
        else:
            assert (not solver.is_dispatcher) and \
                solver.is_worker

Terminating a Solve Early
-------------------------

A solve that is launched without the use `mpiexec` can be
terminated at any point by entering `Ctrl-C` (sending the
process a `SIGINT` signal). If the signal is successfully
received, the solver will attempt to gracefully stop the
solve after it finishes processing the current node, and it
will mark the :attr:`termination_condition
<pybnb.solver.SolverResults.termination_condition>`
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
criteria, starting with the nodes remaining in the queue
from a previous solve. The following code block shows how
this can be done.

.. code-block:: python

    solver = pybnb.Solver()
    solver.solve(problem,
                 node_limit=10)
    queue = solver.save_dispatcher_queue()
    solver.solve(problem,
                 initialize_queue=queue)

For the dispatcher process, the :func:`save_dispatcher_queue
<pybnb.solver.Solver.save_dispatcher_queue>` method returns
an object of type :class:`DispatcherQueueData
<pybnb.dispatcher.DispatcherQueueData>`, which can be
assigned to the `initialize_queue` keyword of the
:func:`solve <pybnb.solver.Solver.solve>` method. For
processes that are not the dispatcher, this function returns
`None`, which is the default value of the `initialize_queue`
keyword.

Saving the Optimal Solution
---------------------------

At this time, the solver does not attempt to track any node
data pertaining to the optimal solution. However, the
following optional problem methods can be used to implement
this kind of functionality:

 - :func:`notify_solve_begins
   <pybnb.problem.Problem.notify_solve_begins>`
 - :func:`notify_new_best_objective_received
   <pybnb.problem.Problem.notify_new_best_objective_received>`
 - :func:`notify_new_best_objective
   <pybnb.problem.Problem.notify_new_best_objective>`
 - :func:`notify_solve_finished
   <pybnb.problem.Problem.notify_solve_finished>`

The code block below shows these methods being used to save
a solution to the Traveling Salesperson Problem. The full
example can be found `here
<https://github.com/ghackebeil/pybnb/blob/master/examples/scripts/tsp/tsp_naive.py>`_.

.. literalinclude:: ../../examples/scripts/tsp/tsp_naive.py
   :language: python
   :lines: 140-180
   :dedent: 4

The code shown above saves the path loaded by the most
recent call to :func:`load_state
<pybnb.problem.Problem.load_state>` when the solver
identifies it as a new best
(:func:`notify_new_best_objective
<pybnb.problem.Problem.notify_new_best_objective>`). Then,
when the solve ends, it is determined which process is
storing the optimal tour so it can be broadcast to everyone
and placed on the results object that will be returned from
the :func:`Solver.solve <pybnb.solver.Solver.solve>` method
(:func:`notify_solve_finished
<pybnb.problem.Problem.notify_solve_finished>`).
