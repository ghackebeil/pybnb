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

Defining a Problem
------------------

To define a branch-and-bound problem with ``pybnb``, one
must define a class that implements the :class:`pybnb.Problem
<pybnb.problem.Problem>` interface, which includes defining
the six required methods shown below.

.. code-block:: python

    import pybnb
    class MyProblem(pybnb.Problem):
       ...
       def sense(self): ...
       def objective(self): ...
       def bound(self): ...
       def save_state(self, node): ...
       def load_state(self, node): ...
       def branch(self, parent_node): ...

.. note::
    The :class:`pybnb.Problem <pybnb.problem.Problem>` base
    class is a purely abstract interface that adds no
    additional data to a problem implementation. It is not
    required to call ``Problem.__init__`` when defining the
    ``__init__`` method on a derived class.

The remainder of this section includes a detailed
description for each of these methods.

 - :func:`Problem.sense() <pybnb.problem.Problem.sense>`

   This is the easiest method to define for a
   branch-and-bound problem. It should return the objective
   sense of the problem, which should always be one of
   :obj:`pybnb.minimize <pybnb.common.minimize>` or
   :obj:`pybnb.maximize <pybnb.common.maximize>`, and should
   not change what it returns over the lifetime of a
   problem. For instance, to define a problem with an
   objective value that should be minimized, the
   implementation would look something like:

   .. code-block:: python

       class MyProblem(pybnb.Problem):
          def sense(self):
              return pybnb.minimize

   The :class:`pybnb.Problem <pybnb.problem.Problem>` base
   class defines two additional convenience methods
   :func:`pybnb.Problem.infeasible_objective()
   <pybnb.problem.Problem.infeasible_objective>` and
   :func:`pybnb.Problem.unbounded_objective()
   <pybnb.problem.Problem.unbounded_objective>` that return
   `+inf` or `-inf`, depending on the return value of
   :func:`sense() <pybnb.problem.Problem.sense>`.

 - :func:`Problem.bound() <pybnb.problem.Problem.bound>`

   This method should return a valid bound for the objective
   function over the current problem domain (as defined by
   the current problem state).

 - :func:`Problem.objective() <pybnb.problem.Problem.objective>`

   This method should return a value for the objective
   function that is feasible for the current problem domain
   (as defined by the current problem state), or it can
   return a value that is worse (as defined by the problem
   sense) than any feasible objective for the current
   problem domain (e.g., :func:`self.infeasible_objective()
   <pybnb.problem.Problem.infeasible_objective>`).

 - :func:`Problem.save_state(node) <pybnb.problem.Problem.save_state>`

   This method should save any relevant state information
   about the problem into the numeric :attr:`state
   <pybnb.node.Node.state>` array on the node argument. The
   amount of available storage in this array can be adjusted
   by calling the :func:`resize() <pybnb.node.Node.resize>`
   method on the node.

 - :func:`Problem.load_state(node) <pybnb.problem.Problem.load_state>`

   This method should load the problem state stored inside
   the numeric :attr:`state <pybnb.node.Node.state>` array
   on the node argument. For instance, a pair of
   :func:`save_state <pybnb.problem.Problem.save_state>` and
   :func:`load_state <pybnb.problem.Problem.load_state>`
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

 - :func:`Problem.branch(parent_node) <pybnb.problem.Problem.branch>`

   This method should partition the problem domain defined
   within the user state on the `parent_node` object into
   zero or more child states and return them as new node
   objects. A child node should be created by calling
   :func:`parent_node.new_child()
   <pybnb.node.Node.new_child>`. Note that for the branching
   process to make sense in the context of a
   branch-and-bound solve, the bound computed from the child
   node states should improve (or not be worse than) the
   bound for the parent node. Once the child bound is
   computed, if it is found to be worse than the bound from
   the parent node, then the branch-and-bound solver will
   issue a warning, as this is likely indicative of a
   programming error or other numerical issues.

   When this method is called, the :attr:`parent_node.bound
   <pybnb.node.Node.bound>` attribute will have been set to the
   value returned from :func:`Problem.bound()
   <pybnb.problem.Problem.bound>` after having the user
   state stored on the `parent_node` loaded into the
   problem. Any child nodes returned from
   :func:`parent_node.new_child()
   <pybnb.node.Node.new_child>` will inherit this bound,
   which may affect their prioritization in the global work
   queue. To override the default bound associated with a
   child node, simply assign a new value to the :attr:`bound
   <pybnb.node.Node.bound>` attribute.

Solving a Problem
-----------------

Under construction ... (see :func:`pybnb.solve <pybnb.solver.solve>` for now)
