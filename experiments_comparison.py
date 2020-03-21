#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:18:07 2020

@author: Wenqing Hu (Missouri S&T)

Title: Experiment for the paper <<Novel Deep Reinforcement Learning Algorithms applied to Joint Control 
of Manufacturing andOnsite Microgrid System>>

Experiment consists of 
1. Comparison of the total cost, total throughput and total energy demand for the 
optimal policy selected by reinforcement learning and the random policy; 
2. Comparison of the total cost, total throughput and total energy demand for the 
optimal policy selected by reinforcement learning and the routine straregy via mixed-integer programming.
"""

output = open('train_output.txt', 'w')
testoutput = open('test_output.txt', 'w') 
bmoutput = open('benchmark_output.txt', 'w')
rtoutput = open('routine_output.txt', 'w')

from microgrid_manufacturing_system import Microgrid, ManufacturingSystem, ActionSimulation, MicrogridActionSet_Discrete_Remainder, MachineActionTree
from projectionSimplex import projection
from reinforcement_learning import action_value, critic, update_theta
from Simple_Manufacturing_System_routine_strategy import Mixed_Integer_Program

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

number_machines=5

import pandas as pd
#read the solar irradiance and wind speed data from file#
file_SolarIrradiance = "SolarIrradiance.csv"
file_WindSpeed = "WindSpeed.csv"
file_rateConsumptionCharge = "rate_consumption_charge.csv"

data_solar = pd.read_csv(file_SolarIrradiance)
solarirradiance = np.array(data_solar.iloc[:,3])

data_wind = pd.read_csv(file_WindSpeed)
windspeed = np.array(data_wind.iloc[:,3])/1000

data_rate_consumption_charge = pd.read_csv(file_rateConsumptionCharge)
rate_consumption_charge = np.array(data_rate_consumption_charge.iloc[:,4])/10


#the learning rates for the theta and omega iterations#
lr_theta=0.003
lr_omega=0.0003

#the discount factor gamma when calculating the total cost#
gamma=0.999

#number of training and testing iterations#
training_number_iteration=1000
testing_number_iteration=10

#unit reward for each unit of production, r^p#
unit_reward_production=5/10000

"""
Reinforcement Learning Algorithm: Off policy TD control combined with actor-critique
Algorithm 1 in the paper

When optimal policy is found, must add
1. Total cost and throughput in given time horizon that the 
   algorithm is used to guide the bilateral control.
2. Total energy demand across all time periods of the given 
   time horizon and the proportion of the energy supply to satisfy the demand. 
   
Two comparisons are made at the test: 
1. Comparison of the total cost, total throughput and total energy demand for the 
optimal policy selected by reinforcement learning and the random policy; 
2. Comparison of the total cost, total throughput and total energy demand for the 
optimal policy selected by reinforcement learning and the routine straregy via mixed-integer programming.
"""

if __name__ == "__main__":
    tf.enable_eager_execution()
    #K.clear_session()
    grid=Microgrid(workingstatus=[0,0,0],
                   SOC=0,
                   actions_adjustingstatus=[0,0,0],
                   actions_solar=[0,0,0],
                   actions_wind=[0,0,0],
                   actions_generator=[0,0,0],
                   actions_purchased=[0,0],
                   actions_discharged=0,
                   solarirradiance=0,
                   windspeed=0
                   )
    System=ManufacturingSystem(machine_states=["Off" for _ in range(number_machines)],
                               machine_control_actions=["K" for _ in range(number_machines)],
                               buffer_states=[2 for _ in range(number_machines-1)],
                               grid=grid
                               )
    
    #randomly generate an initial theta and plot the bounday of the simplex where theta moves#
    r=np.random.uniform(0,1,size=6)
    
    #initialize the theta variable#
    theta=[r[0]*r[1], r[0]*(1-r[1]), r[2]*r[3], r[2]*(1-r[3]), r[4]*r[5], r[4]*(1-r[5])] 
    thetainit=theta
    
    x = [[0, 0], [0, 1], [1, 0]] 
    y = [[0, 1], [1, 0], [0, 0]]
    plt.figure(figsize = (14,10))
    for i in range(len(x)): 
        plt.plot(x[i], y[i], color='g')
    
    my_critic = critic()
    omega = []
    Q=[]
    TemporalDifference=[]
    rewardseq=[]
    reward=0
    diff_omega=[]
    
    #reinforcement learning training process
    for t in range(training_number_iteration):
        print("---------- iteration", t, "----------")
        #beginning of the iteration loop for reinforcement learning training process
        #current states and actions S_t and A_t are stored in class System
        print("*********************Time Step", t, "*********************", file=output)
        for i in range(number_machines):
            print("Machine", System.machine[i].name, "=", System.machine[i].state, ",", "action=", System.machine[i].control_action, file=output)
            print(" Energy Consumption=", System.machine[i].EnergyConsumption(), file=output)
            if System.machine[i].is_last_machine:
                print(" ", file=output)
                print(" throughput=", System.machine[i].LastMachineProduction(), file=output)
            print(" ", file=output)
            if i!=number_machines-1:
                print("Buffer", System.buffer[i].name, "=", System.buffer[i].state, file=output)
        print("Microgrid working status [solar PV, wind turbine, generator]=", System.grid.workingstatus, ", SOC=", System.grid.SOC, file=output)
        print(" microgrid actions [solar PV, wind turbine, generator]=", System.grid.actions_adjustingstatus, file=output)
        print(" solar energy supporting [manufaturing, charging battery, sold back]=", System.grid.actions_solar, file=output)
        print(" wind energy supporting [manufacturing, charging battery, sold back]=", System.grid.actions_wind, file=output)
        print(" generator energy supporting [manufacturing, charging battery, sold back]=", System.grid.actions_generator, file=output)
        print(" energy purchased from grid supporting [manufacturing, charging battery]=", System.grid.actions_purchased, file=output)
        print(" energy discharged by the battery supporting manufacturing=", System.grid.actions_discharged, file=output)
        print(" solar irradiance=", System.grid.solarirradiance, file=output)
        print(" wind speed=", System.grid.windspeed, file=output)
        print(" Microgrid Energy Consumption=", System.grid.EnergyConsumption(), file=output)
        print(" Microgrid Operational Cost=", System.grid.OperationalCost(), file=output)
        print(" Microgrid SoldBackReward=", System.grid.SoldBackReward(), file=output)
        #calculate the total cost at S_t, A_t: E(S_t, A_t)
        E=System.average_total_cost(rate_consumption_charge[t//8640])
        print(" ", file=output)
        print("Average Total Cost=", E, file=output)
        #calculate the old Q-value and its gradient wrt omega: Q(S_t, A_t; omega_t) and grad_omega Q(S_t, A_t; omega_t)
        av=action_value(System,my_critic)
        num_list_SA=av.num_list_States_Actions()
        print("states and actions in numerical list=", np.array(num_list_SA), file=output)
        Q_old=av.Q(num_list_SA)
        Q_grad_omega_old=av.Q_grad_omega(num_list_SA)
        print("Q=", Q_old, file=output)
        Q.append(Q_old)
        #update the theta using deterministic policy gradient#
        upd_tht=update_theta(System, theta)
        print("A_c_grad_theta=", " ", file=output)
        print(upd_tht.A_c_gradient_theta(), file=output)
        print("Q_grad_A_c=", av.Q_grad_A_c(), file=output)
        print("learning rate theta=", lr_theta, file=output)
        policy_grad=upd_tht.deterministic_policygradient(upd_tht.A_c_gradient_theta(), av.Q_grad_A_c())
        print("policy gradient=", policy_grad, file=output)
        theta=upd_tht.update(policy_grad, lr_theta)
        print("new theta=", theta, file=output)
        #calculate the next states and actions: S_{t+1}, A_{t+1}#        
        next_machine_states, next_buffer_states=System.transition_manufacturing()
        next_workingstatus, next_SOC=System.grid.transition()
        next_action=ActionSimulation(System=ManufacturingSystem(machine_states=next_machine_states,
                                                                machine_control_actions=["K" for _ in range(number_machines)],
                                                                buffer_states=next_buffer_states,
                                                                grid=Microgrid(workingstatus=next_workingstatus,
                                                                               SOC=next_SOC,
                                                                               actions_adjustingstatus=[0,0,0],
                                                                               actions_solar=[0,0,0],
                                                                               actions_wind=[0,0,0],
                                                                               actions_generator=[0,0,0],
                                                                               actions_purchased=[0,0],
                                                                               actions_discharged=0,
                                                                               solarirradiance=solarirradiance[t//8640],
                                                                               windspeed=windspeed[t//8640]
                                                                               )
                                                                )
                                    )
        next_actions_adjustingstatus=next_action.MicroGridActions_adjustingstatus()
        next_actions_solar, next_actions_wind, next_actions_generator=next_action.MicroGridActions_SolarWindGenerator(theta)
        next_actions_purchased, next_actions_discharged=next_action.MicroGridActions_PurchasedDischarged(next_actions_solar,
                                                                                                         next_actions_wind,
                                                                                                         next_actions_generator)
        next_machine_control_actions=next_action.MachineActions()
        grid=Microgrid(workingstatus=next_workingstatus,
                       SOC=next_SOC,
                       actions_adjustingstatus=next_actions_adjustingstatus,
                       actions_solar=next_actions_solar,
                       actions_wind=next_actions_wind,
                       actions_generator=next_actions_generator,
                       actions_purchased=next_actions_purchased,
                       actions_discharged=next_actions_discharged,
                       solarirradiance=solarirradiance[t//8640],
                       windspeed=windspeed[t//8640]
                       )
        System=ManufacturingSystem(machine_states=next_machine_states, 
                                   machine_control_actions=next_machine_control_actions, 
                                   buffer_states=next_buffer_states,
                                   grid=grid
                                   )        
        
        #TD-control SARSA#
        av=action_value(System, my_critic)
        num_list_SA=av.num_list_States_Actions()
        Q_new=av.Q(num_list_SA)
        TD=E+gamma*Q_new-Q_old
        print("TD=", TD, file=output)
        TemporalDifference.append(TD)
        #calculate the up-to-date reward#
        reward=reward+np.power(gamma, t)*E
        rewardseq.append(reward)
        print("cumulative reward=", reward, file=output)
        #update omega using actor-critique#
        print("Q_grad_omega=", Q_grad_omega_old, file=output)
        factor=lr_omega*TD
        print("lr_omega*TD=", factor, file=output)
        print("omega=", np.array(omega), file=output)
        my_critic.update_weights(factor)
        #update and calculate the norm difference in omega#
        if t==0:
            diff_omega.append(0)
            omega = [var.numpy() for var in my_critic.trainable_variables]
        else:
            omega_old=omega
            omega = [var.numpy() for var in my_critic.trainable_variables]
            omega_new=omega
            diff_omega.append(np.sum([np.linalg.norm(new_var-old_var) for new_var, old_var in zip(omega_new,omega_old)]))
       
        #discount the learning rate#
        lr_theta=lr_theta*1
        lr_omega=lr_omega*0.999
        
        print(" ", file=output)
        
        #plot the theta values#
        plt.scatter(theta[0], theta[1], color='b')
        plt.scatter(theta[2], theta[3], marker='+', color='m')
        plt.scatter(theta[4], theta[5], marker='*', color='r')
        #end of the iteration loop for reinforcement learning training process#
    
    #plot the theta dynamics#
    plt.savefig('theta.png')
    plt.show()  
       
    #plot the Q values#
    plt.figure(figsize = (14,10))
    plt.plot(Q)
    plt.xlabel('iteration')
    plt.ylabel('action-value-function')
    plt.savefig('Q.png')
    plt.show() 
    
    #plot the temporal differences#
    plt.figure(figsize = (14,10))
    plt.plot(TemporalDifference)
    plt.xlabel('iteration')
    plt.ylabel('temporal difference function')
    plt.savefig('TD.png')
    plt.show()   
    
    #plot the reward sequences#
    plt.figure(figsize = (14,10))
    plt.plot(rewardseq)
    plt.xlabel('iteration')
    plt.ylabel('Sum of rewards during episode')
    plt.savefig('rewards.png')
    plt.show()       

    #plot the weight difference sequences#
    plt.figure(figsize = (14,10))
    plt.plot(diff_omega)
    plt.xlabel('iteration')
    plt.ylabel('L2 norm of the difference in the weights')
    plt.savefig('weightdifference.png')
    plt.show()   
    
    
    #initialize the sequence for the total cost, throughput, energy demand for 
    #the optimal control, the random control (benchmark) and the mixed-integer programming routine strategy#
    totalcostlist_optimal=[0]
    totalthroughputlist_optimal=[0]
    totalenergydemandlist_optimal=[0]
    
    totalcostlist_benchmark=[0]
    totalthroughputlist_benchmark=[0]
    totalenergydemandlist_benchmark=[0]

    totalcostlist_routine=[0]
    totalthroughputlist_routine=[0]
    totalenergydemandlist_routine=[0]

    """
    The 1st Comparision Test: Comparison of the total cost, total throughput and total energy demand for the 
        optimal policy selected by reinforcement learning and the random policy; 
    """
    
    #with the optimal theta and optimal omega at hand, run the system at a certain time horizon#
    #output the optimal theta and optimal omega#
    thetaoptimal=theta
    omegaoptimal=omega   

    print("***Run the system on optimal policy at a time horizon=", testing_number_iteration,"***", file=testoutput)
    print("\n", file=testoutput)
    print("initial proportion of energy supply=", thetainit, file=testoutput)
    print("optimal proportion of energy supply=", thetaoptimal, file=testoutput)
    print("optimal parameter for the neural-network integrated action-value function=", omegaoptimal, file=testoutput)
    print("\n", file=testoutput)
    #run the MDP under optimal theta and optimal omega#
    #at every step search among all discrete actions to find A^d_*=argmin_{A^d}Q(A^d, A^c(thetaoptimal), A^r(A^d, A^c(thetaoptimal)))#
    #Calculate 1. Total cost (E) and throughput in given time horizon that the algorithm is used to guide the bilateral control#
    #Calculate 2. Total energy demand across all time periods of the given time horizon#

    #initialize the system and the grid#
    grid=Microgrid(workingstatus=[0,0,0],
                   SOC=0,
                   actions_adjustingstatus=[0,0,0],
                   actions_solar=[0,0,0],
                   actions_wind=[0,0,0],
                   actions_generator=[0,0,0],
                   actions_purchased=[0,0],
                   actions_discharged=0,
                   solarirradiance=0,
                   windspeed=0
                   )
    System=ManufacturingSystem(machine_states=["Off" for _ in range(number_machines)],
                               machine_control_actions=["K" for _ in range(number_machines)],
                               buffer_states=[2 for _ in range(number_machines-1)],
                               grid=grid
                               )

    #set the total cost, total throughput and the total energy demand#
    totalcost=0
    totalthroughput=0
    totalenergydemand=0 
    target_output=0
    #reinforcement learning testing loop
    for t in range(testing_number_iteration):
        #start of the iteration loop for reinforcement learning training process#
        #current states and actions S_t and A_t are stored in class System#
        #Print (S_t, A_t)#
        print("*********************Time Step", t, "*********************", file=testoutput)
        for i in range(number_machines):
            print("Machine", System.machine[i].name, "=", System.machine[i].state, ",", "action=", System.machine[i].control_action, file=testoutput)
            print(" Energy Consumption=", System.machine[i].EnergyConsumption(), file=testoutput)
            if System.machine[i].is_last_machine:
                print("\n", file=testoutput)
                print(" throughput=", System.machine[i].LastMachineProduction(), file=testoutput)
            print("\n", file=testoutput)
            if i!=number_machines-1:
                print("Buffer", System.buffer[i].name, "=", System.buffer[i].state, file=testoutput)
        print("Microgrid working status [solar PV, wind turbine, generator]=", System.grid.workingstatus, ", SOC=", System.grid.SOC, file=testoutput)
        print(" microgrid actions [solar PV, wind turbine, generator]=", System.grid.actions_adjustingstatus, file=testoutput)
        print(" solar energy supporting [manufaturing, charging battery, sold back]=", System.grid.actions_solar, file=testoutput)
        print(" wind energy supporting [manufacturing, charging battery, sold back]=", System.grid.actions_wind, file=testoutput)
        print(" generator energy supporting [manufacturing, charging battery, sold back]=", System.grid.actions_generator, file=testoutput)
        print(" energy purchased from grid supporting [manufacturing, charging battery]=", System.grid.actions_purchased, file=testoutput)
        print(" energy discharged by the battery supporting manufacturing=", System.grid.actions_discharged, file=testoutput)
        print(" solar irradiance=", System.grid.solarirradiance, file=testoutput)
        print(" wind speed=", System.grid.windspeed, file=testoutput)
        print(" Microgrid Energy Consumption=", System.grid.EnergyConsumption(), file=testoutput)
        print(" Microgrid Operational Cost=", System.grid.OperationalCost(), file=testoutput)
        print(" Microgrid SoldBackReward=", System.grid.SoldBackReward(), file=testoutput)
        #accumulate the total throughput#
        totalthroughput+=System.throughput()
        totalthroughputlist_optimal.append(totalthroughput)
        target_output+=System.throughput()/unit_reward_production
        #calculate the total cost at S_t, A_t: E(S_t, A_t)#
        E=System.average_total_cost(rate_consumption_charge[t//8640])
        print("\n", file=testoutput)
        print("Average Total Cost=", E, file=testoutput)
        #accumulate the total cost#
        totalcost+=E
        totalcostlist_optimal.append(totalcost)
        #accumulate the total energy demand#
        totalenergydemand+=System.energydemand(rate_consumption_charge[t//8640])
        totalenergydemandlist_optimal.append(totalenergydemand)
        #determine the next system and grid states#
        next_machine_states, next_buffer_states=System.transition_manufacturing()
        next_workingstatus, next_SOC=System.grid.transition()
        #determine the next continuous actions A^c(thetaoptimal)=energy distribued to [solar, wind, generator]#
        #auxiliarysystem with next system and grid states#
        AuxiliarySystem=ManufacturingSystem(machine_states=next_machine_states,
                                            machine_control_actions=["K" for _ in range(number_machines)],
                                            buffer_states=next_buffer_states,
                                            grid=Microgrid(workingstatus=next_workingstatus,
                                                           SOC=next_SOC,
                                                           actions_adjustingstatus=[0,0,0],
                                                           actions_solar=[0,0,0],
                                                           actions_wind=[0,0,0],
                                                           actions_generator=[0,0,0],
                                                           actions_purchased=[0,0],
                                                           actions_discharged=0,
                                                           solarirradiance=solarirradiance[t//8640],
                                                           windspeed=windspeed[t//8640]
                                                           )
                                            )
        #under the next system and grid states, calculate the energy generated by the solar PV, e_t^s; the wind turbine, e_t^w; the generator, e_t^g#
        energy_generated_solar=AuxiliarySystem.grid.energy_generated_solar()
        energy_generated_wind=AuxiliarySystem.grid.energy_generated_wind()
        energy_generated_generator=AuxiliarySystem.grid.energy_generated_generator()
        #under the optimal theta, calculate the optimal continuous actions A^c(thetaoptimal)=energy distributed in [solar, wind, generator]#
        optimal_next_actions_solar=[energy_generated_solar*thetaoptimal[1-1], energy_generated_solar*thetaoptimal[2-1], energy_generated_solar*(1-thetaoptimal[1-1]-thetaoptimal[2-1])]
        optimal_next_actions_wind=[energy_generated_wind*thetaoptimal[3-1], energy_generated_wind*thetaoptimal[4-1], energy_generated_wind*(1-thetaoptimal[3-1]-thetaoptimal[4-1])]
        optimal_next_actions_generator=[energy_generated_generator*thetaoptimal[5-1], energy_generated_generator*thetaoptimal[6-1], energy_generated_generator*(1-thetaoptimal[5-1]-thetaoptimal[6-1])]
        #bulid the list of the set of all admissible machine actions#    
        machine_action_tree=MachineActionTree(machine_action="ROOT")
        machine_action_tree.BuildTree(AuxiliarySystem, level=0, tree=machine_action_tree)
        machine_action_list=[]
        machine_action_tree.TraverseTree(level=0, tree=machine_action_tree, machine_action_list=[])
        machine_action_set_list=machine_action_tree.machine_action_set_list
        #build the list of the set of all admissible microgrid actions for adjusting the status and for purchase/discharge
        microgrid_action_set_DR=MicrogridActionSet_Discrete_Remainder(AuxiliarySystem)
        microgrid_action_set_list_adjustingstatus=microgrid_action_set_DR.List_AdjustingStatus()
        microgrid_action_set_list_purchased_discharged=microgrid_action_set_DR.List_PurchasedDischarged(actions_solar=optimal_next_actions_solar,
                                                                                                        actions_wind=optimal_next_actions_wind,
                                                                                                        actions_generator=optimal_next_actions_generator)
        #determine the next discrete/remainder actions by finding A^{*,d}_{t+1}=argmin_{A^d}Q(S_{t+1}, A^d, A^c(thetaoptimal), A^r(A^d, A^c(thetaoptimal)); omegaoptimal)#
        #output A^{*,d}_{t+1}, A^{*,r}_{t+1}=A^r(A^{*,d}_{t+1}, A^c(thetaoptimal))
        #output Q^*_{t+1}=Q(S_{t+1}, A^{*,d}_{t+1}, A^c(thetaoptimal), A^{*,r}_{t+1}; omegaoptimal)
        optimal_Q=0
        optimal_next_machine_actions=[]
        optimal_next_microgrid_actions_adjustingstatus=[]
        optimal_next_microgrid_actions_purchased=[]
        optimal_next_microgrid_actions_discharged=0
        i=1
        for machine_action_list in machine_action_set_list:
            for microgrid_action_list_adjustingstatus in microgrid_action_set_list_adjustingstatus:
                for microgrid_action_list_purchased_discharged in microgrid_action_set_list_purchased_discharged:
                    AuxiliarySystem=ManufacturingSystem(machine_states=next_machine_states,
                                                        machine_control_actions=machine_action_list,
                                                        buffer_states=next_buffer_states,
                                                        grid=Microgrid(workingstatus=next_workingstatus,
                                                                       SOC=next_SOC,
                                                                       actions_adjustingstatus=microgrid_action_list_adjustingstatus,
                                                                       actions_solar=optimal_next_actions_solar,
                                                                       actions_wind=optimal_next_actions_wind,
                                                                       actions_generator=optimal_next_actions_generator,
                                                                       actions_purchased=microgrid_action_list_purchased_discharged[0],
                                                                       actions_discharged=microgrid_action_list_purchased_discharged[1],
                                                                       solarirradiance=solarirradiance[t//8640],
                                                                       windspeed=windspeed[t//8640]
                                                                       )
                                                        )
                    av=action_value(AuxiliarySystem, my_critic)
                    num_list_SA=av.num_list_States_Actions()
                    Q=av.Q(num_list_SA)
                    if i==1:
                        optimal_Q=Q
                        optimal_next_machine_actions=machine_action_list
                        optimal_next_microgrid_actions_adjustingstatus=microgrid_action_list_adjustingstatus
                        optimal_next_microgrid_actions_purchased=microgrid_action_list_purchased_discharged[0]
                        optimal_next_microgrid_actions_discharged=microgrid_action_list_purchased_discharged[1]
                    else:
                        if Q<optimal_Q:
                            optimal_Q=Q
                            optimal_next_machine_actions=machine_action_list
                            optimal_next_microgrid_actions_adjustingstatus=microgrid_action_list_adjustingstatus
                            optimal_next_microgrid_actions_purchased=microgrid_action_list_purchased_discharged[0]
                            optimal_next_microgrid_actions_discharged=microgrid_action_list_purchased_discharged[1]
                    i=i+1
                    
        #update the manufacturing system and the grid according to S_{t+1}, A^{*,d}_{t+1}, A^c(thetaoptimal), A^{*,r}_{t+1}#
        grid=Microgrid(workingstatus=next_workingstatus,
                       SOC=next_SOC,
                       actions_adjustingstatus=optimal_next_microgrid_actions_adjustingstatus,
                       actions_solar=optimal_next_actions_solar,
                       actions_wind=optimal_next_actions_wind,
                       actions_generator=optimal_next_actions_generator,
                       actions_purchased=optimal_next_microgrid_actions_purchased,
                       actions_discharged=optimal_next_microgrid_actions_discharged,
                       solarirradiance=solarirradiance[t//8640],
                       windspeed=windspeed[t//8640]
                       )
        System=ManufacturingSystem(machine_states=next_machine_states, 
                                   machine_control_actions=optimal_next_machine_actions, 
                                   buffer_states=next_buffer_states,
                                   grid=grid
                                   )        
        #end of the iteration loop for reinforcement learning training process#
    print("total cost=", totalcost, file=testoutput)    
    print("total throughput=", totalthroughput, file=testoutput)    
    print("total energy demand=", totalenergydemand, file=testoutput)
        
    
    #As benchmark, with initial theta and randomly simulated actions, run the system at a certain time horizon#
    print("\n*************************BenchMark System with initial theta and random actions*************************", file=bmoutput)
    print("***Run the system on random policy at a time horizon=", testing_number_iteration,"***", file=bmoutput)
    print("\n", file=bmoutput)
    print("initial proportion of energy supply=", thetainit, file=bmoutput)
    print("\n", file=bmoutput)
    #initialize the system and the grid#
    grid=Microgrid(workingstatus=[0,0,0],
                   SOC=0,
                   actions_adjustingstatus=[0,0,0],
                   actions_solar=[0,0,0],
                   actions_wind=[0,0,0],
                   actions_generator=[0,0,0],
                   actions_purchased=[0,0],
                   actions_discharged=0,
                   solarirradiance=0,
                   windspeed=0
                   )
    System=ManufacturingSystem(machine_states=["Off" for _ in range(number_machines)],
                               machine_control_actions=["K" for _ in range(number_machines)],
                               buffer_states=[2 for _ in range(number_machines-1)],
                               grid=grid
                               )
    #set the total cost, total throughput and the total energy demand#
    totalcost=0
    totalthroughput=0
    totalenergydemand=0 
    #benchmark system iteration loop
    for t in range(testing_number_iteration):
        #start of the iteration loop for a benchmark system with initial theta and random actions#
        #current states and actions S_t and A_t are stored in class System#
        #Pring (S_t, A_t)#
        print("*********************Time Step", t, "*********************", file=bmoutput)
        for i in range(number_machines):
            print("Machine", System.machine[i].name, "=", System.machine[i].state, ",", "action=", System.machine[i].control_action, file=bmoutput)
            print(" Energy Consumption=", System.machine[i].EnergyConsumption(), file=bmoutput)
            if System.machine[i].is_last_machine:
                print("\n", file=bmoutput)
                print(" throughput=", System.machine[i].LastMachineProduction(), file=bmoutput)
            print("\n", file=bmoutput)
            if i!=number_machines-1:
                print("Buffer", System.buffer[i].name, "=", System.buffer[i].state, file=bmoutput)
        print("Microgrid working status [solar PV, wind turbine, generator]=", System.grid.workingstatus, ", SOC=", System.grid.SOC, file=bmoutput)
        print(" microgrid actions [solar PV, wind turbine, generator]=", System.grid.actions_adjustingstatus, file=bmoutput)
        print(" solar energy supporting [manufaturing, charging battery, sold back]=", System.grid.actions_solar, file=bmoutput)
        print(" wind energy supporting [manufacturing, charging battery, sold back]=", System.grid.actions_wind, file=bmoutput)
        print(" generator energy supporting [manufacturing, charging battery, sold back]=", System.grid.actions_generator, file=bmoutput)
        print(" energy purchased from grid supporting [manufacturing, charging battery]=", System.grid.actions_purchased, file=bmoutput)
        print(" energy discharged by the battery supporting manufacturing=", System.grid.actions_discharged, file=bmoutput)
        print(" solar irradiance=", System.grid.solarirradiance, file=bmoutput)
        print(" wind speed=", System.grid.windspeed, file=bmoutput)
        print(" Microgrid Energy Consumption=", System.grid.EnergyConsumption(), file=bmoutput)
        print(" Microgrid Operational Cost=", System.grid.OperationalCost(), file=bmoutput)
        print(" Microgrid SoldBackReward=", System.grid.SoldBackReward(), file=bmoutput)
        #accumulate the total throughput#
        totalthroughput+=System.throughput()
        totalthroughputlist_benchmark.append(totalthroughput)
        #calculate the total cost at S_t, A_t: E(S_t, A_t)#
        E=System.average_total_cost(rate_consumption_charge[t//8640])
        print("\n", file=bmoutput)
        print("Average Total Cost=", E, file=bmoutput)
        #accumulate the total cost#
        totalcost+=E
        totalcostlist_benchmark.append(totalcost)
        #accumulate the total energy demand#
        totalenergydemand+=System.energydemand(rate_consumption_charge[t//8640])
        totalenergydemandlist_benchmark.append(totalenergydemand)
        #determine the next system and grid states#
        next_machine_states, next_buffer_states=System.transition_manufacturing()
        next_workingstatus, next_SOC=System.grid.transition()
        next_action=ActionSimulation(System=ManufacturingSystem(machine_states=next_machine_states,
                                                                machine_control_actions=["K" for _ in range(number_machines)],
                                                                buffer_states=next_buffer_states,
                                                                grid=Microgrid(workingstatus=next_workingstatus,
                                                                               SOC=next_SOC,
                                                                               actions_adjustingstatus=[0,0,0],
                                                                               actions_solar=[0,0,0],
                                                                               actions_wind=[0,0,0],
                                                                               actions_generator=[0,0,0],
                                                                               actions_purchased=[0,0],
                                                                               actions_discharged=0,
                                                                               solarirradiance=solarirradiance[t//8640],
                                                                               windspeed=windspeed[t//8640]
                                                                               )
                                                                )
                                    )
        next_actions_solar, next_actions_wind, next_actions_generator=next_action.MicroGridActions_SolarWindGenerator(thetainit)
        next_actions_purchased, next_actions_discharged=next_action.MicroGridActions_PurchasedDischarged(next_actions_solar,
                                                                                                         next_actions_wind,
                                                                                                         next_actions_generator)
        next_machine_control_actions=next_action.MachineActions()
        #update the manufacturing system and the grid according to S_{t+1}, A^{*,d}_{t+1}, A^c(thetaoptimal), A^{*,r}_{t+1}#
        grid=Microgrid(workingstatus=next_workingstatus,
                       SOC=next_SOC,
                       actions_adjustingstatus=next_actions_adjustingstatus,
                       actions_solar=next_actions_solar,
                       actions_wind=next_actions_wind,
                       actions_generator=next_actions_generator,
                       actions_purchased=next_actions_purchased,
                       actions_discharged=next_actions_discharged,
                       solarirradiance=solarirradiance[t//8640],
                       windspeed=windspeed[t//8640]
                       )
        System=ManufacturingSystem(machine_states=next_machine_states, 
                                   machine_control_actions=next_machine_control_actions, 
                                   buffer_states=next_buffer_states,
                                   grid=grid
                                   )        
        #end of the iteration loop for for a benchmark system with initial theta and random actions#
    print("total cost=", totalcost, file=bmoutput)    
    print("total throughput=", totalthroughput, file=bmoutput)   
    print("total energy demand=", totalenergydemand, file=bmoutput)
    
    """
    The 2nd Comparision Test: Comparison of the total cost, total throughput and total energy demand for the 
        optimal policy selected by reinforcement learning and the routine strategy selected by the mixed-integer programming;        
    """
    #Calculate and output the total cost, total throughput and total energy demand for mixed-integer programming with target output as the one given by the optimal strategy
    print("\n************************* Mixed Integer Programming with given Target Output *************************", file=rtoutput)
    print("***Run the system on routine policy by mixed-integer programming at a time horizon=", testing_number_iteration,"***", file=rtoutput)
    target_output=int(target_output)
    print("Target Output =", target_output, file=rtoutput)
    routine_sol=Mixed_Integer_Program(target_output)
    print("Optimal solution from mixed-integer programming is given by \n", routine_sol.T, file=rtoutput)
    
    
    
    #plot and compare the total cost, the total throughput and the total energy demand for optimal control and random control (benchmark)#
    #plot the total cost#
    plt.figure(figsize = (14,10))
    plt.plot(totalcostlist_optimal, '-', color='r')
    plt.plot(totalcostlist_benchmark, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total cost')
    plt.title('Total cost under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalcost.png')
    plt.show()  

    #plot the total throughput#
    plt.figure(figsize = (14,10))
    plt.plot(totalthroughputlist_optimal, '-', color='r')
    plt.plot(totalthroughputlist_benchmark, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total throughput')
    plt.title('Total throughput under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalthroughput.png')
    plt.show()  

    #plot the total energy demand#
    plt.figure(figsize = (14,10))
    plt.plot(totalenergydemandlist_optimal, '-', color='r')
    plt.plot(totalenergydemandlist_benchmark, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total energy demand')
    plt.title('Total energy demand under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalenergydemand.png')
    plt.show()  
    
output.close() 
testoutput.close()
bmoutput.close()
rtoutput.close()