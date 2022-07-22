# RL-manufacturing
Source code for the paper 

Joint Control of Manufacturing and Onsite Microgrid System via Novel Neural-Network Integrated Reinforcement Learning Algorithms

by Yang, J., Sun, Z., Hu, W. and Steimeister, L.

Accepted at <i>Applied Energy</i>.

<b>The paper with Supplementary Materials is available here as the file MDP_paper_20220220_AppliedEnergyERevise.docx</b>

The run files are 

1. <b>experiments_comparison.py</b>

  compares the efficiency of optimal solution selected by reinforcement learning, by mixed-integer programming routine          strategy and by benchmark random policy.
  
2. <b>mip_plot.ipynb, plot_average_experiments.ipynb</b>

  plot the comparison of total energy cost and total production throughput in units for the optimal policy and mixed-integer programming policy; also plot the average over 3 times of these experiments.


The main files are

3. <b>microgrid_manufacturing_system.py</b>

  simulates the joint operation of microgrid and manufacturing system.

4. <b>reinforcement_learning.py</b>

  reinforcement learning via two layer fully connected neural network. 

5. <b>Simple_Manufacturing_System-Pure_Q-Learning.py, 1st_on.npy, 2nd_on.npy, both_off.npy, both_on.npy</b>

  learn the microgrid-manufacturing system using pure Q-learning. This is to compare with our new method.

6. <b>Simple_Manufacturing_System_routine_strategy.py</b>

  learn the microgrid-manufacturing system using routine strategy via linear mixed-integer programming.
  
7. <b>mip-solver.xlsx</b>

  solving the mixed-integer programming total cumulative energy cost and total production units given the mixed-integer programming solution.


The auxiliary files are

8. <b>projectionSimplex.py</b>

  proximal operator to the simplex D^c={(x_1, x_2), 0\leq x_i\leq 1, x_1+x_2\leq 1}.

9. <b>SolarIrradiance.csv, WindSpeed.csv, rate_consumption_charge.csv</b>

  1 year data in 8640 hours (360 days * 24 hours) for solar irradiance, wind speed and rate of consumption charge.

10. <b>real-case parameters-experimental-use.xlsx</b>
  
  the scaled real-case parameters for the manufacturing system and the microgrid used in the experiment.
