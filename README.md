# RL-manufacturing
Source code for reinforcement learning algorithms applied to joint control of manufacturing and onsite microgrid system. 

The main files are

1. <b>microgrid_manufacturing_system.py</b>

simulates the joint operation of microgrid and manufacturing system.

2. <b>reinforcement_learning.py</b>

reinforcement learning via two layer fully connected neural network.

3. <b>Simple_Manufacturing_System-Pure_Q-Learning.py</b>

Learn the microgrid-manufacturing system using pure Q-learning. This is to compare with our new method.

4. <b>Simple_Manufacturing_System_routine_strategy.py</b>

Learn the microgrid-manufacturing system using routine strategy via linear mixed-integer programming.

The auxiliary files are

5. <b>projectionSimplex.py</b>

proximal operator to the simplex D^c={(x_1, x_2), 0\leq x_i\leq 1, x_1+x_2\leq 1}.

6. <b>SolarIrradiance.csv, WindSpeed.csv, rate_consumption_charge.csv</b>

1 year data in 8640 hours (360 days * 24 hours) for solar irradiance, wind speed and rate of consumption charge.

