# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:16:27 2020

@author: Louis Steimeister (Missouri S&T)
"""

from scipy import optimize as opt
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

dt = 1
num_machines = 5
time_horizon = 10
capacity_of_buffer = [2]*(num_machines-1)
rated_power_of_machine = [2.5]*num_machines
production_rate_of_machine = [1]*num_machines
target_output = 7
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
print("Delta shape: ", DELTA.shape)
ZERO        =  sp.sparse.csr_matrix(DELTA.shape)
B1_descr    =  [["D" if i<=j else "Z" for i in range(num_machines)] for j in range(num_machines)]
B1_mat_vec  =  [[DELTA if i<=j else ZERO for i in range(time_horizon)] for j in range(time_horizon)]
B1          =  sp.sparse.bmat(B1_mat_vec)
B2          = -B1
C1          =  np.array([capacity_of_buffer for i in range(time_horizon)]).flatten()
C2          =  np.zeros(B2.shape[0])
print("BufferMat:", B1.todense())
print("1 shape: ", B1.shape, C1.shape)
print("2 shape: ", B2.shape, C2.shape)
del B1_mat_vec
print(B1_descr)
###########################################
# Production Condition
B3          = sp.sparse.eye(num_machines*time_horizon)
B4          = -B3
C3          = np.ones(num_machines*time_horizon)
C4          = np.zeros(num_machines*time_horizon)
###########################################
# Minimal Production Condition
B5          = -sp.sparse.hstack([sp.sparse.csr_matrix((1,(num_machines-1)*time_horizon)),np.ones((1,time_horizon))])
C5          = -np.array([target_output])
###########################################
# Finalize Conditions
#print([B1,B2,B3,B4,B5])
B           = sp.sparse.vstack([B1,B2,B3,B4,B5])
C           = np.concatenate([C1,C2,C3,C4,C5])
#print(B, "dim: ", B.shape)
#print(C, "dim: ", C.shape)

###########################################
# Formulate Minimization
# min! Ax

A           = np.transpose(np.array(rated_power_of_machine*time_horizon))*dt
                          
res         = opt.linprog(c=A,A_ub=B,b_ub = C)                
prod_mat    = np.round(np.array(res.x).reshape((num_machines,time_horizon)),decimals=5)
#print(prod_mat)
print("output is:",-B5 * res.x )
print("Buffer is:",B1 * res.x )

            

fig, ax = plt.subplots(num_machines,1, figsize=(10,20))
for k, a in enumerate(ax):
  a.plot(prod_mat[k,:])
  # plt.plot()