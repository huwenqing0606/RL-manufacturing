# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:16:27 2020
"""

import numpy as np

"""
*************************************Linear and Mixed-Integer Programming*************************************
@author: Louis Steimeister (Missouri S&T)
"""
from scipy import optimize as opt
import scipy as sp
from matplotlib import pyplot as plt

dt = 1
num_machines = 5
time_horizon = 100
capacity_of_buffer = [3]*(num_machines-1)
rated_power_of_machine = [2.5]*num_machines
production_rate_of_machine = [1]*num_machines
target_output = 10
###########################################
# Set up Condition
# Bx <= C
###########################################
# Buffer Condition
DELTA       =  (sp.sparse.hstack([sp.sparse.diags(production_rate_of_machine[0:(num_machines-1)]),
                                  sp.sparse.csr_matrix((num_machines-1,1))])
                -sp.sparse.hstack([sp.sparse.csr_matrix((num_machines-1,1)),
                                   sp.sparse.diags(production_rate_of_machine[1:num_machines])
                                   ]))
#print("Delta shape: ", DELTA.shape)
ZERO        =  sp.sparse.csr_matrix(DELTA.shape)
B1_descr    =  [["D" if i<=j else "Z" for i in range(num_machines)] for j in range(num_machines)]
B1_mat_vec  =  [[DELTA if i<=j else ZERO for i in range(time_horizon)] for j in range(time_horizon)]
B1          =  sp.sparse.bmat(B1_mat_vec)
B2          = -B1
C1          =  np.array([capacity_of_buffer for i in range(time_horizon)]).flatten()
C2          =  np.zeros(B2.shape[0])
#print("BufferMat:", B1.todense())
#print("BufferMat >= 0:", B2.todense())
#print("1 shape: ", B1.shape, C1.shape)
#print("2 shape: ", B2.shape, C2.shape)
del B1_mat_vec
#print(B1_descr)
###########################################
# Production Condition
#B3          = sp.sparse.eye(num_machines*time_horizon)
#B4          = -B3
#C3          = np.ones(num_machines*time_horizon)
#C4          = np.zeros(num_machines*time_horizon)
###########################################
# Minimal Production Condition
B5  = -np.concatenate([np.array([0]*(num_machines-1)+[1]) for _ in range(time_horizon)]).reshape((1,num_machines*time_horizon))
C5          = -np.array([target_output])
###########################################
# Finalize Conditions
#print([B1,B2,B3,B4,B5])
#B           = sp.sparse.vstack([B1,B2,B3,B4,B5])
#C           = np.concatenate([C1,C2,C3,C4,C5])
B           = sp.sparse.vstack([B1,B2,B5])
C           = np.concatenate([C1,C2,C5])
#print(B, "dim: ", B.shape)
#print(C, "dim: ", C.shape)

###########################################
# Linear programming
# Formulate Minimization
# min! Ax

A           = np.transpose(np.array(rated_power_of_machine*time_horizon))*dt
Bounds = np.hstack([np.zeros((num_machines*time_horizon,1)),
                    np.ones((num_machines*time_horizon,1))])                         
res         = opt.linprog(c=A,A_ub=B,b_ub = C,bounds=Bounds,options = {"maxiter": 100000, "rr": False})                
prod_mat    = np.round(np.array(res.x).reshape((num_machines,time_horizon),order = "F"),decimals=5)
#print(prod_mat)
#print("output is:",np.round(-B5 * res.x,5))
#print("Buffer is:",np.round(B1 * res.x ,5))

            

#fig, ax = plt.subplots(num_machines,1, figsize=(10,20))
#for k, a in enumerate(ax):
#  a.plot(prod_mat[k,:])
#  a.set_ylim(-.01,1)
#  a.axhline(0)
#  # plt.plot()

#Mixed integer programming
import mip

# set up optimization model
m = mip.Model()
# create decision/optimization variables
# x = "Production" in {0,1}
x = [m.add_var(var_type=mip.BINARY) for _ in range(time_horizon*num_machines)]
#print(x)

#define matrix multiplication which outputs linear combinations of the optimization variable
def mipMatMult(Mat,Vec):
  if isinstance(Mat,sp.sparse.coo_matrix):
    my_MAT = Mat.tocsr()
  else: my_MAT = Mat
  out=[]
  for i in range(Mat.shape[0]):
    temp = mip.entities.LinExpr()
    for j in range(Mat.shape[1]):
      if my_MAT[i,j] != 0:
          temp += my_MAT[i,j]*x[j]
    out.append(temp)
  return out

# add constraints to the optimization model
ineq_constraint_lst = mipMatMult(B,x)
for k in range(B.shape[0]):
  m += ineq_constraint_lst[k]<=C[k]

# define objective function
objective = mip.entities.LinExpr()
for k in range(len(x)):
  if C[k]!=0:
    objective += C[k]*x[k]
m.objective= mip.minimize(objective)

# run optimization
m.integer_tol = .0001
m.start = [(x[k], 1.0) for k in range(len(x))]
status = m.optimize()
if status == mip.OptimizationStatus.OPTIMAL:
  print('optimal solution cost {} found'.format(m.objective_value))
elif status == mip.OptimizationStatus.FEASIBLE:
  print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
  print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
  print('solution:')
  for v in m.vars:
    if abs(v.x) > 1e-6:# only printing non-zeros
      print('{} : {}'.format(v.name, v.x))



m.num_solutions

prod_vec    = np.array([v.x for v in m.vars])
prod_mat    = np.array(prod_vec).reshape((num_machines,time_horizon),order = "F")

# compute output
#print("Output")
#print(B5*prod_vec)

# compute Buffers
Buffer_vec = B1*prod_vec+1
Buffer_mat = np.array(Buffer_vec).reshape((num_machines-1,time_horizon),order = "F")
#print("Buffer")
#print(Buffer_mat)


# plots
#fig, ax = plt.subplots(num_machines,2, figsize=(10,10),facecolor=(1,1,1))
#for k, a in enumerate(ax):
#  # plot production
#  a[0].step(y=prod_mat[k,:],x =range(time_horizon))
#  a[0].set_ylim(-.01,1.01)
#  a[0].set_xticks(range(time_horizon))
#  a[0].set_title(f"Prduction of Machine {k}")
#  # plot Buffers
#  if k == num_machines-1: break
#  a[1].step(y=Buffer_mat[k,:],x =range(time_horizon))
#  a[1].set_ylim(1-.01,np.max(capacity_of_buffer)+.01)
#  a[1].set_xticks(range(time_horizon))
#  a[1].set_title(f"Buffer of Machine {k}")
  
#range(time_horizon)

#print("Optimal Production Matrix is ", prod_mat)


"""
*************************************Simple manufacturing system under routine strategy*************************************
@author: Wenqing Hu (Missouri S&T)

Output the total cost, total throughput and total energy demand for simple manufacturing system under routine strategy
These quantities are calculated according to the sheme in our paper
"""

#Set up all parameters that are constant throughout the system

Delta_t=1
#the actual time measured in one decision epoch unit, in hours#
unit_reward_production=5/100
#the unit reward for each unit of production, i.e. the r^p, this applies to the end of the machine sequence#
cutin_windspeed=3/100
#the cut-in windspeed (m/s), v^ci#
cutoff_windspeed=11/100
#the cut-off windspeed (m/s), v^co#
rated_windspeed=7/100
#the rated windspeed (m/s), v^r#
charging_discharging_efficiency=0.95/100
#the charging-discharging efficiency, eta#
rate_battery_discharge=2/100
#the rate for discharging the battery, b#
rate_consumption_charge=0.25/100
#the rate of consumption charge, r^c#
unit_operational_cost_solar=0.17/100
#the unit operational and maintanance cost for generating power from solar PV, r_omc^s#
unit_operational_cost_wind=0.08/100
#the unit operational and maintanance cost for generating power from wind turbine, r_omc^w#
unit_operational_cost_generator=0.45/100
#the unit opeartional and maintanance cost for generating power from generator, r_omc^g#
unit_operational_cost_battery=0.9/100
#the unit operational and maintanance cost for battery storage system per unit charging/discharging cycle, r_omc^b#
capacity_battery_storage=20/100
#the capacity of battery storage system, e#
SOC_max=0.95*2.71828*100000000000
#the maximum state of charge of battery system#
SOC_min=0.05*2.71828/100
#the minimum state of charge of battery system#
area_solarPV=14000/100
#the area of the solar PV system, a#
efficiency_solarPV=0.2/100
#the efficiency of the solar PV system, delta#
density_of_air=1.225/100
#calculate the rated power of the wind turbine, density of air, rho#
radius_wind_turbine_blade=25/100
#calculate the rated power of the wind turbine, radius of the wind turbine blade, r#
average_wind_speed=0.25/100
#calculate the rated power of the wind turbine, average wind speed, v_avg#
power_coefficient=0.593/100
#calculate the rated power of the wind turbine, power coefficient, theta#
gearbox_transmission_efficiency=0.9/100
#calculate the rated power of the wind turbine, gearbox transmission efficiency, eta_t#
electrical_generator_efficiency=0.9/100
#calculate the rated power of the wind turbine, electrical generator efficiency, eta_g#
rated_power_wind_turbine=878.1101/100
#rated_power_wind_turbine=0.5*density_of_air*np.pi*radius_wind_turbine_blade*radius_wind_turbine_blade*average_wind_speed*average_wind_speed*average_wind_speed*power_coefficient*gearbox_transmission_efficiency*electrical_generator_efficiency/1000
#the rated power of the wind turbine, RP_w#
number_windturbine=1/100
#the number of wind turbine in the onsite generation system, N_w#
number_generators=1/100
#the number of generators, n_g#
rated_output_power_generator=650/100
#the rated output power of the generator, G_p#
unit_reward_production=5/100
#unit reward for each unit of production, r^p#
unit_reward_soldbackenergy=0.2/100
#the unit reward from sold back energy, r^sb#
number_machines=5
#the total number of machines in the manufacturing system, total number of buffers=number_machines-1#
testing_number_iteration=100
#number of testing iterations#

#set the optimal production matrx which is a 0-1 matrix, rows=number_machines, columns=testing_number_iteration
x=prod_mat.T
print(x)