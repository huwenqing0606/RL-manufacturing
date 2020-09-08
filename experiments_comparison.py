# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:24:35 2020

@author: Wenqing Hu and Louis Steinmeister (Missouri S&T)
"""

from microgrid_manufacturing_system import SystemInitialize
from reinforcement_learning import Reinforcement_Learning_Training, Reinforcement_Learning_Testing, Benchmark_RandomAction_Testing, Guassian_RL_Training, RL_Gaussian_Testing
from Simple_Manufacturing_System_routine_strategy import RoutineStrategy_Testing

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()




#set the number of machines
number_machines=5
#set the unit reward of production
unit_reward_production=10000/10000
#the unit reward for each unit of production (10^4$/unit produced), i.e. the r^p, this applies to the end of the machine sequence#

#the initial learning rates for the theta and omega iterations#
lr_theta_initial=0.003
lr_omega_initial=0.0003

#number of training and testing iterations#
training_number_iteration=5000
testing_number_iteration=100

#set the seeds for tensorflow, numpy and for Gaussian sampling
tf_seed=2
np_seed = 1 
g_seed = 1

#set the id
ID = ""

#set seeds
tf.random.set_random_seed(tf_seed)    
np.random.seed(np_seed)
    
#set the initial machine states, machine control actions and buffer states
initial_machine_states=["Opr" for _ in range(number_machines)]
initial_machine_actions=["K" for _ in range(number_machines)]
initial_buffer_states=[100 for _ in range(number_machines-1)]
    
#initialize the system
System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)
      
#randomly generate an initial theta and plot the bounday of the simplex where theta moves#
r=np.random.uniform(0,1,size=6)
    
#initialize the theta variable#
theta=[r[0]*r[1], r[0]*(1-r[1]), r[2]*r[3], r[2]*(1-r[3]), r[4]*r[5], r[4]*(1-r[5])] 
#record the initial theta applied before training
thetainit=theta
    
        
x = [[0, 0], [0, 1], [1, 0]] 
y = [[0, 1], [1, 0], [0, 0]]
plt.figure(figsize = (14,10))
for i in range(len(x)): 
    plt.plot(x[i], y[i], color='g')
    
##############################################################################################
##########      TRAINING                                                            #########
##############################################################################################
        
#initialize the system for DPG RL Training
System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)
    
theta, omega, my_critic = Reinforcement_Learning_Training(System, 
                                                          thetainit, 
                                                          lr_theta_initial, 
                                                          lr_omega_initial, 
                                                          training_number_iteration,
                                                          ID = ID)
    
    
#with the optimal theta and optimal omega at hand, run the system at a certain time horizon#
#output the optimal theta and optimal omega#
thetaoptimal=theta
omegaoptimal=omega  
my_critic_optimal=my_critic
    
    
#initialize the system for GPG RL Training
System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)
    
# gaussian training 
gaussian_actor, gaussian_omega, gaussian_critic, gaussian_thetainit = Guassian_RL_Training(System, 
                                                                                           lr_theta_initial, 
                                                                                           lr_omega_initial, 
                                                                                           training_number_iteration,
                                                                                           rand_seed=g_seed,
                                                                                           ID = ID
                                                                                           )
    
    
##############################################################################################
###########      TESTING                                                             #########
##############################################################################################
    
    
    
    
#initialize the system for deterministic policy gradients RL Testing
System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)
    
totalcostlist_optimal, totalthroughputlist_optimal, totalenergydemandlist_optimal, RL_target_output = Reinforcement_Learning_Testing(System, 
                                                                                                                                     thetainit, 
                                                                                                                                     thetaoptimal, 
                                                                                                                                     omegaoptimal, 
                                                                                                                                     my_critic_optimal, 
                                                                                                                                     testing_number_iteration, 
                                                                                                                                     unit_reward_production,
                                                                                                                                     ID = ID
                                                                                                                                     )
    
#initialize the system for gaussian policy gradients RL Testing
System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)
    
totalcostlist_optimal_gaussian, totalthroughputlist_optimal_gaussian, totalenergydemandlist_optimal_gaussian, RL_target_output_gaussian = RL_Gaussian_Testing(System,  
                                                                                                                                                              gaussian_actor,
                                                                                                                                                              gaussian_thetainit,
                                                                                                                                                              omegaoptimal, 
                                                                                                                                                              my_critic_optimal, 
                                                                                                                                                              testing_number_iteration, 
                                                                                                                                                              unit_reward_production,
                                                                                                                                                              ID = ID)
    
    

#As benchmark, with initial theta and randomly simulated actions, run the system at a certain time horizon#
    
#initialize the system
System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)
    
totalcostlist_benchmark, totalthroughputlist_benchmark, totalenergydemandlist_benchmark, random_target_output = Benchmark_RandomAction_Testing(System, 
                                                                                                                                               thetainit, 
                                                                                                                                               testing_number_iteration, 
                                                                                                                                               unit_reward_production,
                                                                                                                                               ID = ID
                                                                                                                                               )
    
    
#if target output is not enough, simply quit, else, continue with comparison and further experiments
if RL_target_output<0*random_target_output:
    print("Not enough production! Test Ended Without Plotting the Comparison...")
else:    
    #plot and compare the total cost, the total throughput and the total energy demand for optimal control and random control (benchmark)#
    #plot the total cost#
    plt.figure(figsize = (14,10))
    plt.plot([value*10000 for value in totalcostlist_optimal], '-', color='r')
    plt.plot([value*10000 for value in totalcostlist_benchmark], '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total cost ($)')
    plt.title('Total cost under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalcost.png')
    plt.show()  

    #plot the total throughput, in dollar amount#
    plt.figure(figsize = (14,10))
    plt.plot([value*10000 for value in totalthroughputlist_optimal], '-', color='r')
    plt.plot([value*10000 for value in totalthroughputlist_benchmark], '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total throughput ($)')
    plt.title('Total throughput under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalthroughput.png')
    plt.show()  
    
    #plot the total throughput, in production units#
    plt.figure(figsize = (14,10))
    plt.plot([value/unit_reward_production for value in totalthroughputlist_optimal], '-', color='r')
    plt.plot([value/unit_reward_production for value in totalthroughputlist_benchmark], '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total throughput (production unit)')
    plt.title('Total throughput (production unit) under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalthroughput_unit.png')
    plt.show()  

    #plot the total energy demand#
    plt.figure(figsize = (14,10))
    plt.plot([value*10000 for value in totalenergydemandlist_optimal], '-', color='r')
    plt.plot([value*10000 for value in totalenergydemandlist_benchmark], '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total energy cost ($)')
    plt.title('Total energy cost under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalenergycost.png')
    plt.show()  
    
    
    """
    The 2nd Comparision Test: Comparison of the total cost, total throughput and total energy demand for the 
        optimal policy selected by reinforcement learning and the routine strategy selected by the mixed-integer programming;        
    """
    target_output=int(RL_target_output)

    RoutineStrategy_Testing(testing_number_iteration, target_output)




# compare our method using deterministic policy gradient on theta and deterministic policy gradient on mean and variance of theta under Gaussian sampling
doCompareGaussian = 1
if doCompareGaussian:
    #plot and compare the total cost, the total throughput and the total energy demand for DPG and DPG-Gaussian
    #plot the total cost#
    plt.figure(figsize = (14,10))
    plt.plot([value*10000 for value in totalcostlist_optimal], '-', color='r',label = "DPG-deterministic policy")
    plt.plot([value*10000 for value in totalcostlist_optimal_gaussian], '-.', color='m', label = "DPG-Gaussian policy")
    plt.xlabel('iteration')
    plt.ylabel('total cost ($)')
    plt.title('Total cost under optimal deterministic policy (red, solid) and optimal gaussian policy (pink, dot-dashed)')
    plt.legend(loc="upper left")
    plt.savefig(f"totalcost_gaussian{ID}.png")
    plt.show()  
    
    #plot the total throughput, in dollar amount#
    plt.figure(figsize = (14,10))
    plt.plot([value*10000 for value in totalthroughputlist_optimal], '-', color='r',label = "det. policy")
    plt.plot([value*10000 for value in totalthroughputlist_optimal_gaussian], '-.', color='m', label = "gauss. policy")
    plt.xlabel('iteration')
    plt.ylabel('total throughput ($)')
    plt.title('Total throughput under optimal deterministic policy (red, solid) and optimal gaussian policy (pink, dot-dashed)')
    plt.legend(loc="upper left")
    plt.savefig(f'totalthroughput_gaussian{ID}.png')
    plt.show()  
    
    #plot the total throughput, in production units#
    plt.figure(figsize = (14,10))
    plt.plot([value/unit_reward_production for value in totalthroughputlist_optimal], '-', color='r',label = "det. policy")
    plt.plot([value/unit_reward_production for value in totalthroughputlist_optimal_gaussian], '-.', color='m', label = "gauss. policy")
    plt.xlabel('iteration')
    plt.ylabel('total throughput (production unit)')
    plt.title('Total throughput (production unit) under optimal deterministic policy (red, solid) and optimal gaussian policy (pink, dot-dashed)')
    plt.legend(loc="upper left")
    plt.savefig(f'totalthroughput_unit_gaussian{ID}.png')
    plt.show()  
        
    #plot the total energy demand#
    plt.figure(figsize = (14,10))
    plt.plot([value*10000 for value in totalenergydemandlist_optimal], '-', color='r',label = "det. policy")
    plt.plot([value*10000 for value in totalenergydemandlist_optimal_gaussian], '-.', color='m', label = "gauss. policy")
    plt.xlabel('iteration')
    plt.ylabel('total energy cost ($)')
    plt.title('Total energy cost under optimal deterministic policy (red, solid) and optimal gaussian policy (pink, dot-dashed)')
    plt.legend(loc="upper left")
    plt.savefig(f'totalenergycost_gaussian{ID}.png')
    plt.show()  
    
    