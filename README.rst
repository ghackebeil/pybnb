pybnb
=====

|PyPI-Status| |PyPI-Versions| |Codacy-Grade|

|Travis-Status| |Appveyor-Status| |Coverage-Status| |Documentation-Status|

A parallel branch-and-bound engine for Python. (https://pybnb.readthedocs.io)

This software is copyright (c) by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).

This software is released under the MIT software license.
This license, including disclaimer, is available in the 'LICENSE' file.

Quick Start
-----------

**Define a problem:**

.. code:: python

  # simple.py

  import pybnb
  class Simple(pybnb.Problem):
      def __init__(self):
          self.bounds = [0.0,1.0]
      def sense(self):
          return pybnb.minimize
      def objective(self):
          return round(self.bounds[1] - self.bounds[0], 3)
      def bound(self):
          return -(self.bounds[1] - self.bounds[0])**2
      def save_state(self, node):
          node.state = self.bounds
      def load_state(self, node):
          self.bounds = node.state
      def branch(self):
          L, U = self.bounds
          mid = 0.5 * (L + U)
          for l,u in [(L,mid), (mid,U)]:
              child = pybnb.Node()
              child.state = (l,u)
              yield child

**Write a solve script:**

.. code:: python

  # solve_simple.py

  import simple
  problem = simple.Simple()
  results = pybnb.solve(problem,
                        absolute_gap=1e-9)

**Run the script:**

.. code:: bash

  $ mpirun -np 4 python solve_simple.py

  Using non-default solver options:
   - absolute_gap: 1e-09 (default: 0)

  Starting branch & bound solve:
   - dispatcher pid: 34902 (Ozymandias.local)
   - worker processes: 3
  ---------------------------------------------------------------------------------------------------------------------------
           Nodes        |                      Objective Bounds                        |              Work              
        Expl    Unexpl  |      Incumbent           Bound     Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
           0         1  |            inf            -inf          inf%             inf |      0.0       0.00     0.00%      0
  *        1         2  |              1              -1  200.0000000%               2 |      0.0    1226.99   300.00%      1
  *        2         3  |            0.5              -1  150.0000000%             1.5 |      0.0    2966.04   150.00%      0
  *        4         5  |           0.25           -0.25   50.0000000%             0.5 |      0.0    8081.95    75.00%      0
  *        8         9  |          0.125         -0.0625   18.7500000%          0.1875 |      0.0   12566.90    37.50%      0
        Expl    Unexpl  |      Incumbent           Bound     Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
  *       16        17  |          0.062       -0.015625    7.7625000%        0.077625 |      0.0   15352.74    18.75%      0
  *       32        33  |          0.031     -0.00390625    3.4906250%      0.03490625 |      0.0   15981.49    18.75%      0
  *       64        65  |          0.016   -0.0009765625    1.6976563%    0.0169765625 |      0.0   18740.68    18.75%      0
  *      128       129  |          0.008   -0.0002441406    0.8244141%  0.008244140625 |      0.0   21573.51    11.72%      0
  *      256       257  |          0.004   -6.103516e-05    0.4061035%  0.004061035156 |      0.0   22166.96     8.20%      0
        Expl    Unexpl  |      Incumbent           Bound     Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
  *      512       513  |          0.002   -1.525879e-05    0.2015259%  0.002015258789 |      0.0   21177.00     5.86%      0
  *     1024      1025  |          0.001   -3.814697e-06    0.1003815%  0.001003814697 |      0.1   19978.42     9.38%      0
  *     2048      2049  |              0   -9.536743e-07    0.0000954% 9.536743164e-07 |      0.1   21606.45     5.42%      0
       24029     24030  |              0   -1.490116e-08    0.0000015% 1.490116119e-08 |      1.1   21961.03     5.98%      0
       46159     46160  |              0    -3.72529e-09    0.0000004% 3.725290298e-09 |      2.1   22120.75     5.73%      0
        Expl    Unexpl  |      Incumbent           Bound     Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
       65537     65538  |              0   -9.313226e-10    0.0000001% 9.313225746e-10 |      3.0   22459.50     6.20%      0
  ---------------------------------------------------------------------------------------------------------------------------
  
  Absolute optimality tolerance met
  Optimal solution found!
  
  solver results:
   - solution_status: optimal
   - termination_condition: optimality
   - objective: 0
   - bound: -9.313226e-10
   - absolute_gap: 9.313226e-10
   - relative_gap: 9.313226e-10
   - nodes: 65537
   - wall_time: 2.96 s
   - best_node: Node(objective=0)
  
  Number of Workers:        3
  Load Imbalance:       6.20%
   - min: 21355 (proc rank=3)
   - max: 22710 (proc rank=1)
  Average Worker Timing:
   - queue:      80.78% [avg time: 109.6 us, count: 65537]
   - load_state:  0.44% [avg time: 596.1 ns, count: 65537]
   - bound:       0.59% [avg time: 796.1 ns, count: 65537]
   - objective:   3.52% [avg time:   4.7 us, count: 65537]
   - branch:      3.36% [avg time:   4.6 us, count: 65537]
   - other:      11.31% [avg time:  15.3 us, count: 65537]


.. |Travis-Status| image:: https://travis-ci.org/ghackebeil/pybnb.svg?branch=master
  :target: https://travis-ci.org/ghackebeil/pybnb
.. |Appveyor-Status| image::  https://ci.appveyor.com/api/projects/status/gfbrxja9v08rm7a2?svg=true
  :target: https://ci.appveyor.com/project/ghackebeil/pybnb
.. |Coverage-Status| image:: https://codecov.io/gh/ghackebeil/pybnb/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/ghackebeil/pybnb
.. |Documentation-Status| image:: https://readthedocs.org/projects/pybnb/badge/?version=latest
  :target: http://pybnb.readthedocs.io/en/latest/?badge=latest
.. |PyPI-Status| image:: https://img.shields.io/pypi/v/pybnb.svg
  :target: https://pypi.python.org/pypi/pybnb/
.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/pybnb.svg
   :target: https://pypi.org/project/pybnb
.. |Codacy-Grade| image:: https://img.shields.io/lgtm/grade/python/g/ghackebeil/pybnb.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/ghackebeil/pybnb/context:python
