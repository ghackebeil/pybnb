Changelog
=========

0.5.0 - `master`_
~~~~~~~~~~~~~~~~~

* Tracking the best node and storing it on the results
  object (#1, #3, #7)
* Removing node argument from `branch` method of Problem
  implementations. This makes it more apparent that
  branching occurs for the node from the most recent
  `load_state` and not some arbitrary node. Child nodes
  should be created using `pybnb.Node()` as opposed to
  `node.new_child()`.
* Removing `tree_id` and `parent_tree_id` attributes from
  the Node class.
* Adding a helper class for performing a nested
  branch-and-bound solve and adding command-line options
  that allow it to be used with the command-line examples.

0.4.0 - 2019-01-31
~~~~~~~~~~~~~~~~~~

* Major redesign of node serialization, allowing for much
  faster serial performance (especially with PyPy): The
  `node.state` attribute can now be assigned anything (must
  pickle-able to work in the parallel case). The
  `node.resize(...)` method should no longer be called. (#5)
* Adding `pybnb.config` to allow customizing serialization
  of node state (e.g., use `dill` instead of `pickle`). See
  online docs for more information.
* Allowing for lexicographic queue strategies: The
  `queue_strategy` solve option can now be assigned a tuple
  of existing strategies (e.g., `('bound','objective')`) to
  define a lexicographic queue strategy. Equivalently, when
  the 'custom' queue strategy is selected, users can assign
  a tuple to the `node.queue_priority` attribute.
* Adding a last-in, first-out queue strategy ('lifo').

0.3.0 - 2019-01-20
~~~~~~~~~~~~~~~~~~

* Adding new sections to the online docs.
* Fix to pybnb.solve helper function. The dispatcher_rank
  keyword was not being used to initialized the solver.
* Adding 'log_new_incumbent' solver option that can be set
  to False to disable immediate logging of incumbent
  updates, in case this is slowing down the dispatcher.
* Renaming 'node_priority_strategy' solver option to
  'queue_strategy'.
* Adding 'scale_function' solver option to allow customizing
  how the absolute gap is converted to a relative
  gap. Default is max{1.0,|objective|}.
* Renaming 'absolute_tolerance' solver option to
  'comparison_tolerance'.
* Renaming the pybnb.pyomo_tools sub-package to pybnb.pyomo.
* Renaming the 'cutoff' solve option to 'bound_stop'. Also
  adding an 'objective_stop' solve option. Both options
  trigger the 'objective_limit' termination condition
  (replaces the 'cutoff' termination condition).
* Adding an optional notify_solve_begins callback to the
  Problem base class.
* Removing worker_comm argument from the
  notify_new_best_objective_* callbacks, as it is unsafe to
  use when those methods are called.
* Adding documented Enum types for solution status,
  termination condition, and node priority strategy.
* Adding handling for SIGINT and SIGUSER1 events to
  initiate a graceful shutdown that tags the results
  with an 'interrupted' termination condition.
* Adding 'disable_objective_call' solver option that
  disables calls to the problem objective.
* Removing a number of collective MPI communicator calls
  from the solve finalization process.
* Adding additional queue strategy implementations: 'random'
  and 'local_gap'.

0.2.9 - 2018-12-03
~~~~~~~~~~~~~~~~~~

* tagging with explicit support for Python 3.7 on PyPI

0.2.8 - 2018-11-26
~~~~~~~~~~~~~~~~~~

* removing code that causes deprecation warnings in Python 3.7

0.2.7 - 2018-11-26
~~~~~~~~~~~~~~~~~~

* removing code that causes deprecation warnings in Python 3.7
* compatibility fixes for next pyomo release

0.2.6 - 2018-07-13
~~~~~~~~~~~~~~~~~~

* adding a best objective first node priority strategy
* performance improvements

0.2.5 - 2018-05-30
~~~~~~~~~~~~~~~~~~

* performance improvements

0.2.4 - 2018-05-26
~~~~~~~~~~~~~~~~~~

* adding first-in, first-out dispatcher queue strategy
* changing how solver handles single-process communicators
* removed unnecessary Barrier from solve method
* fixed bug in pyomo_tools that left lingering send calls
* documentation improvements

0.2.3 - 2018-05-20
~~~~~~~~~~~~~~~~~~

* Initial working release on PyPI.

.. _`master`: https://github.com/ghackebeil/pybnb
