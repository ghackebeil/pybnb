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
       def branch(self): ...
       # optional methods
       def notify_solve_begins(self,
                               comm,
                               worker_comm,
                               convergence_checker):
           ...
       def notify_new_best_node(self,
                                node,
                                current):
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
   about the problem onto the :attr:`state
   <pybnb.node.Node.state>` attribute of node argument. If
   one wishes to utilize the MPI-based parallel solver, the
   only requirement for what goes into the node state is
   that it can be serialized using the ``pickle`` or
   ``dill`` modules. By default, ``pybnb`` is configured to
   use the ``pickle`` module for node serialization. See the
   section titled :ref:`configuration` for details on how to
   adjust this and related settings.

 - :func:`Problem.load_state(node) <pybnb.problem.Problem.load_state>`

   This method should load the problem state stored on the
   :attr:`state <pybnb.node.Node.state>` attribute of the
   node argument. The code block below shows an example pair
   of :func:`save_state <pybnb.problem.Problem.save_state>`
   and :func:`load_state <pybnb.problem.Problem.load_state>`
   implementations.

   .. code-block:: python

       class MyProblem(pybnb.Problem):
           def __init__(self):
               self._L = 0.0
               self._U = 1.0
           def save_state(self, node):
               node.state = (self._L, self._U)
           def load_state(self, node):
               (self._L, self._U) = node.state

 - :func:`Problem.branch() <pybnb.problem.Problem.branch>`

   This method should partition the problem domain defined
   by the current user state into zero or more child states
   and return them on new nodes. A child node can be created
   by directly instantiating a :class:`pybnb.Node
   <pybnb.node.Node>` object. Note that for the branching
   process to make sense, the bound computed from the child
   states should improve (or not be worse than) the bound
   for their parent node. Once the child bound is computed,
   the solver will issue a warning if it is found to be
   worse than the bound from its parent node, as this is
   indicative of a programming error or other numerical
   issues.

   Note that any child nodes returned from
   :func:`Problem.branch() <pybnb.problem.Problem.branch>`
   will automatically be assigned the bound and objective
   from their parent for potential use in determining their
   prioritization in the global work queue. Users can
   override this by manually assigning a value to one or
   both of these node attributes before yielding them from
   the branch method.

   Additionally, further control over the prioritization of
   a child node can be achieved by setting the
   `queue_strategy` solve option to "custom", and then
   directly assigning a value to the :attr:`queue_priority
   <pybnb.node.Node.queue_priority>` attribute of the child
   node before it is yielded.

Solving a Problem
-----------------
There are two approaches to solving a branch-and-bound
problem with ``pybnb``. The first is to simply call the
:func:`solve <pybnb.solver.solve>` convenience
function. This will create a :class:`Solver
<pybnb.solver.Solver>` object, call the :func:`Solver.solve
<pybnb.solver.Solver.solve>` method, and report the results
as well as additional timing information about the solve.

.. code-block:: python

    import pybnb
    problem = MyProblem()
    results = pybnb.solve(problem,
                          relative_gap=1e-4)

The second approach is to manually create a :class:`Solver
<pybnb.solver.Solver>` object and call the
:func:`Solver.solve <pybnb.solver.Solver.solve>` method
directly.

Both approaches can solve a problem in serial or
parallel. The difference is that the :func:`solve
<pybnb.solver.solve>` convenience function provides a few
additional options that simplify the process of saving
solver output and results to a file. Additionally,
collecting the timing information reported by this function
adds some additional communication overhead to the end of
the solve; thus, the second approach of directly using a
:class:`Solver <pybnb.solver.Solver>` can be more efficient.

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
``ImportError``. The optional keywords `comm` and
`dispatcher_rank` can be used to change the default
behavior.

When a solver is created with ``Solver(comm=None)``, this
will disable any attempted import of ``mpi4py``, allowing
problems to be solved without the use of any parallel
functionality. The `comm` keyword can also be assigned a
communicator different from ``mpi4py.MPI.COMM_WORLD``. If
the solver communicator includes more than one process, the
`dispatcher_rank` keyword can be assigned a process rank
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

How the Solver Calls the Problem Methods
----------------------------------------

The following block of pseudocode provides a high-level
overview of how the solver calls the methods on a
user-defined problem. Highlighted lines show where problem
methods are called.

.. code-block:: python
  :emphasize-lines: 5,6,8,16,17,18,19,21,23,24,26,31,32
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
            node, best_node = dispatcher.update(...)
            if <conditional_1>:
               problem.notify_new_best_node(node=best_node,
                                            current=False)
            problem.load_state(node)
            bound = problem.bound()
            if <conditional_2>:
                objective = problem.objective()
                if <conditional_3>:
                    problem.notify_new_best_node(node=node,
                                                 current=True)
                if <conditional_4>:
                    children = problem.branch()

        #
        # solve finalization
        #
        problem.load_state(root)
        problem.notify_solve_finished(...)

Note that during the main solve loop (starting on line 13),
it is safe to assume that the six highlighted problem
methods between line 13 and line 25 will be called in the
relative order shown. The conditions under which these
methods will be called are briefly discussed below:

 - **<conditional_1>** (line 15): This condition is met when
   the `best_node` received from the dispatcher is not
   unbounded and improves upon the best node currently known
   to the worker process (i.e., has a better objective). By
   default, the check for objective improvement is exact,
   but it can be relaxed by assigning a nonzero value to the
   `comparison_tolerance` keyword of the :func:`Solver.solve
   <pybnb.solver.Solver.solve>` method.
 - **<conditional_2>** (line 20): This condition is met when
   the bound computed by the problem for the current node
   makes it eligible for the queue relative to the best
   objective known to the process. By default, this is true
   when the bound is better than the best objective by any
   nonzero amount, but this behavior can be influenced using
   the `queue_tolerance` keyword of the :func:`Solver.solve
   <pybnb.solver.Solver.solve>` method.
 - **<conditional_3>** (line 22): This condition is met when
   the objective computed by the problem for the current
   node is not unbounded and improves upon the objective of
   the best node currently known to the process. By default,
   the check for improvement is exact, but it can be relaxed
   by assigning a nonzero value to the
   `comparison_tolerance` keyword of the :func:`Solver.solve
   <pybnb.solver.Solver.solve>` method.
 - **<conditional_4>** (line 25): This condition is met when
   the objective computed by the problem for the current
   node is not unbounded, when **<conditional_2>** is still
   satisfied (based on a potentially new best objective),
   and when the difference between the node's updated bound
   and objective satisfies the branching tolerance. By
   default, the branching tolerance is zero, meaning that
   any nonzero distance between these two values will
   satisfy this check, but this can be adjusted using the
   `branching_tolerance` keyword of the :func:`Solver.solve
   <pybnb.solver.Solver.solve>` method.
