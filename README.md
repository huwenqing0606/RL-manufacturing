# RL-manufacturing
Source code for reinforcement learning algorithms applied to joint control of manufacturing and onsite microgrid system. 

The run file is 

1. <b>experiments_comparison.py</b>

compares the efficiency of optimal solution selected by reinforcement learning, by mixed-integer programming routine strategy and by benchmark random policy.

The main files are

2. <b>microgrid_manufacturing_system.py</b>

simulates the joint operation of microgrid and manufacturing system.

3. <b>reinforcement_learning.py</b>

reinforcement learning via two layer fully connected neural network.

4. <b>Simple_Manufacturing_System-Pure_Q-Learning.py</b>

Learn the microgrid-manufacturing system using pure Q-learning. This is to compare with our new method.

5. <b>Simple_Manufacturing_System_routine_strategy.py</b>

Learn the microgrid-manufacturing system using routine strategy via linear mixed-integer programming.

The auxiliary files are

6. <b>projectionSimplex.py</b>

proximal operator to the simplex D^c={(x_1, x_2), 0\leq x_i\leq 1, x_1+x_2\leq 1}.

7. <b>SolarIrradiance.csv, WindSpeed.csv, rate_consumption_charge.csv</b>

1 year data in 8640 hours (360 days * 24 hours) for solar irradiance, wind speed and rate of consumption charge.

