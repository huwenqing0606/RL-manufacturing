# RL-manufacturing
Source code for reinforcement learning algorithms applied to joint control of manufacturing and onsite microgrid system. 

<i>The main files are </i>

1. microgrid_manufacturing_system.py 

simulates the joint operation of microgrid and manufacturing system

2. reinforcement_learning.py

reinforcement learning via two layer fully connected neural network

3. Simple_Manufacturing_System-Pure_Q-Learning.py

Learn the microgrid-manufacturing system using pure Q-learning. This is to compare with our new method

4. Simple_Manufacturing_System_routine_strategy.py

routine strategy via linear programming

<i>The auxiliary files are </i>

5. projectionSimplex.py

proximal operator to the simplex D^c={(x_1, x_2), 0\leq x_i\leq 1, x_1+x_2\leq 1}.

