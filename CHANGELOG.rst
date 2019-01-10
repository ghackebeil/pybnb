Changelog
=========

0.3.1 - `master`_
~~~~~~~~~~~~~~~~~

* TODO

0.3.0 - 2019-01-10
~~~~~~~~~~~~~~~~~~

* Renaming the pybnb.pyomo_tools sub-package to pybnb.pyomo.
* Renaming the cutoff solve option to bound_stop. Also
  adding an objective_stop solve option. Both options
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
* Adding a solver option that disables calls to the problem
  objective.
* Removing a number of collective MPI communicator calls
  from the solve finalization process.
* Adding a queue implementation that uses random priority
  assignments.

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
