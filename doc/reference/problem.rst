pybnb.problem
-------------

.. automodule:: pybnb.problem

  .. autoclass:: pybnb.problem.Problem
    :members: infeasible_objective, unbounded_objective, notify_solve_begins, notify_new_best_node, notify_solve_finished

    .. automethod:: pybnb.problem.Problem.sense
      :abstractmethod:

    .. automethod:: pybnb.problem.Problem.bound
      :abstractmethod:

    .. automethod:: pybnb.problem.Problem.objective
      :abstractmethod:

    .. automethod:: pybnb.problem.Problem.save_state
      :abstractmethod:

    .. automethod:: pybnb.problem.Problem.load_state
      :abstractmethod:

    .. automethod:: pybnb.problem.Problem.branch
      :abstractmethod:
