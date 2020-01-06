# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:33:36 2020

@author: huwenqing

Title: Reinforcement Learning for the joint control of onsite microgrid and manufacturing system
"""

from microgrid_manufacturing_system import Microgrid, ManufacturingSystem, ActionSimulation
from projectionSimplex import projection
import numpy as np
import matplotlib.pyplot as plt

number_machines=4

import pandas as pd
#read the solar irradiance and wind speed data from file#
file_SolarIrradiance = "SolarIrradiance.csv"
file_WindSpeed = "WindSpeed.csv"

data_solar = pd.read_csv(file_SolarIrradiance)
solarirradiance = np.array(data_solar.iloc[:,3])

data_wind = pd.read_csv(file_WindSpeed)
windspeed = np.array(data_wind.iloc[:,3])


#the learning rates for the theta and omega iterations#
lr_theta=0.01
lr_omega=0.01

#the discount factor gamma#
gamma=0.1


"""
Provide the structure of the action-value function Q(A^d, A^c, A^r; omega), 
also provide its gradients tensor with respect to A^c and to omega
""" 
class action_value(object):
    def __init__(self,
                 System=ManufacturingSystem(machine_states=["Off" for _ in range(number_machines)],
                                            machine_control_actions=["K" for _ in range(number_machines)],
                                            buffer_states=[0 for _ in range(number_machines-1)],
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
                                            ),
                omega=[ [0 for _ in range(number_machines)], 
                        [0 for _ in range(number_machines-1)], 
                        [0 for _ in range(3)],
                        [0 for _ in range(number_machines)],
                        [0 for _ in range(3)],
                        [0 for _ in range(9)],
                        [0 for _ in range(2)],
                        0 ] 
                ):
        self.System=System
        #omega is the weight vector presented as a numerical list#
        self.omega=omega       
        
    def num_list_States_Actions(self):
        #return the states and actions as a numerical list#
        #"Off"=1, "Brk"=2, "Idl"=3, "Blk"=4, "Opr"=5#
        #"H"=1, "K"=2, "W"=3#
        list=[  [0 for _ in range(number_machines)], 
                [0 for _ in range(number_machines-1)], 
                [0 for _ in range(3)],
                [0 for _ in range(number_machines)],
                [0 for _ in range(3)],
                [0 for _ in range(9)],
                [0 for _ in range(2)],
                0 ]
        for i in range(number_machines):
            if self.System.machine_states[i]=="Off":
                list[1-1][i]=1
            elif self.System.machine_states[i]=="Brk":
                list[1-1][i]=2
            elif self.System.machine_states[i]=="Idl":
                list[1-1][i]=3
            elif self.System.machine_states[i]=="Blo":
                list[1-1][i]=4
            else:
                list[1-1][i]=5
        for i in range(number_machines-1):
            list[2-1][i]=self.System.buffer_states[i]
        for i in range(3):
            list[3-1][i]=self.System.grid.workingstatus[i]
        for i in range(number_machines):
            if self.System.machine_control_actions[i]=="H":
                list[4-1][i]=1
            elif self.System.machine_control_actions[i]=="K":
                list[4-1][i]=2
            else:
                list[4-1][i]=3
        for i in range(3):
            list[5-1][i]=self.System.grid.actions_adjustingstatus[i]
        for i in range(3):
            list[6-1][i]=self.System.grid.actions_solar[i]
            list[6-1][i+3]=self.System.grid.actions_wind[i]
            list[6-1][i+6]=self.System.grid.actions_generator[i]
        for i in range(2):
            list[7-1][i]=self.System.grid.actions_purchased[i]
        list[8-1]=self.System.grid.actions_discharged
        return list
        
    
    def Q(self, num_list_States_Actions):
        Q=0
        for i in range(8):
            Q=Q+np.dot(np.array(self.omega[i]), np.array(num_list_States_Actions[i]))    
        return Q
    
    def Q_grad_A_c(self):
        grad=[]
        for i in range(9):
            grad.append(self.omega[6-1][i])
        return np.array(grad)

    def Q_grad_omega(self, num_list_States_Actions):
        return num_list_States_Actions




"""
Provide the necessary functions for deterministic policy gradient updates for the theta
"""
class update_theta(object):
    def __init__(self,
                 System=ManufacturingSystem(machine_states=["Off" for _ in range(number_machines)],
                                            machine_control_actions=["K" for _ in range(number_machines)],
                                            buffer_states=[0 for _ in range(number_machines-1)],
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
                                            ),
                 theta=[0,0,0,0,0,0]
                 ):
        #System has state and action (S_t, A_t), theta corresponds to A_t^c=A_t^c(theta)#
        self.System=System
        self.theta=theta
    
    def A_c_gradient_theta(self):
        #output the gradient tensor of the A^c with respect to the theta variables#
        grad=[[],[],[],[],[],[],[],[],[]]
        #calculate the energy generated by the solar PV, e_t^s#
        energy_generated_solar=self.System.grid.energy_generated_solar()
        #calculate the energy generated by the wind turbine, e_t^w#
        energy_generated_wind=self.System.grid.energy_generated_wind()
        #calculate the energy generated bv the generator, e_t^g#
        energy_generated_generator=self.System.grid.energy_generated_generator()
        grad=[[energy_generated_solar, 0, 0, 0, 0, 0], 
              [0, energy_generated_solar, 0, 0, 0, 0], 
              [-energy_generated_solar, -energy_generated_solar, 0, 0, 0, 0],
              [0, 0, energy_generated_wind, 0, 0, 0],
              [0, 0, 0, energy_generated_wind, 0, 0],
              [0, 0, -energy_generated_wind, -energy_generated_wind, 0, 0],
              [0, 0, 0, 0, energy_generated_generator, 0],
              [0, 0, 0, 0, 0, energy_generated_generator],
              [0, 0, 0, 0, -energy_generated_generator, -energy_generated_generator]]
        return np.array(grad)
        
    def deterministic_policygradient(self, A_c_grad_theta, Q_grad_A_c):
        #output the deterministic policy gradient of the cost with respect to the theta#
        policygradient=np.dot(Q_grad_A_c,A_c_grad_theta)
        return policygradient

    def update(self, policygradient, lr_theta):
        theta_old=self.theta
        theta_new=projection(theta_old-lr_theta*policygradient)                           
        return theta_new




"""
Reinforcement Learning Algorithm: Off policy TD control combined with actor-critique
Algorithm 1 in the paper
"""
if __name__ == "__main__":
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
                               buffer_states=[0 for _ in range(number_machines-1)],
                               grid=grid
                               )
    theta=[0,0,0,0,0,0]
    omega=[ [0 for _ in range(number_machines)], 
            [0 for _ in range(number_machines-1)], 
            [0 for _ in range(3)],
            [0 for _ in range(number_machines)],
            [0 for _ in range(3)],
            [0 for _ in range(9)],
            [0 for _ in range(2)],
            0 ] 
    Q=[]
    for t in range(1000):
        #current states and actions S_t and A_t are stored in class System#
        print("*********************Time Step", t, "*********************")
        for i in range(number_machines):
            print(System.machine[i].PrintMachine())
            if i!=number_machines-1:
                print(System.buffer[i].PrintBuffer())
        print(System.grid.PrintMicrogrid())
        print("Average Total Cost=", System.average_total_cost())
        #calculate the old Q-value and its gradient wrt omega: Q(S_t, A_t; omega_t) and grad_omega Q(S_t, A_t; omega_t)#
        av=action_value(System, omega)
        num_list_SA=av.num_list_States_Actions()
        Q_old=av.Q(num_list_SA)
        Q_grad_omega_old=av.Q_grad_omega(num_list_SA)
        print("Q=", Q_old)
        print(" ")
        Q.append(Q_old)
        #Calculate the total cost at S_t, A_t: E(S_t, A_t)#
        E=System.average_total_cost()
        #update the theta using deterministic policy gradient#
        update_tht=update_theta(System, theta)
        policy_grad=update_tht.deterministic_policygradient(update_tht.A_c_gradient_theta(), av.Q_grad_A_c())
        theta=update_tht.update(policy_grad, lr_theta)
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
                                                                               windspeed=windspeed[t//8640],
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
        av=action_value(System, omega)
        num_list_SA=av.num_list_States_Actions()
        Q_new=av.Q(num_list_SA)
        TD=E+gamma*Q_new-Q_old
        print("TD=", TD)
        #update omega using actor-critique#
        factor=lr_omega*TD
        for i in range(number_machines):
            omega[1-1][i]=omega[1-1][i]-factor*Q_grad_omega_old[1-1][i]
        for i in range(number_machines-1):
            omega[2-1][i]=omega[2-1][i]-factor*Q_grad_omega_old[2-1][i]
        for i in range(3):
            omega[3-1][i]=omega[3-1][i]-factor*Q_grad_omega_old[3-1][i]
        for i in range(number_machines):
            omega[4-1][i]=omega[4-1][i]-factor*Q_grad_omega_old[4-1][i]
        for i in range(3):
            omega[5-1][i]=omega[5-1][i]-factor*Q_grad_omega_old[5-1][i]
        for i in range(3):
            omega[6-1][i]=omega[6-1][i]-factor*Q_grad_omega_old[6-1][i]
            omega[6-1][i+3]=omega[6-1][i+3]-factor*Q_grad_omega_old[6-1][i+3]
            omega[6-1][i+6]=omega[6-1][i+6]-factor*Q_grad_omega_old[6-1][i+6]
        for i in range(2):
            omega[7-1][i]=omega[7-1][i]-factor*Q_grad_omega_old[7-1][i]
        omega[8-1]=omega[8-1]-factor*Q_grad_omega_old[8-1]
        
        
        lr_theta=lr_theta*0.99
        lr_omega=lr_omega*0.99
        
    
    plt.plot(Q)
    plt.show()   