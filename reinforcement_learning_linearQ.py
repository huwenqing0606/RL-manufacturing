# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:33:36 2020

@author: huwenqing

Title: Reinforcement Learning for the joint control of onsite microgrid and manufacturing system
"""
output = open('output.txt', 'w') 

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
lr_omega=0.001

#the discount factor gamma when calculating the total cost#
gamma=0.01


"""
Provide the structure of the action-value function Q(S, A^d, A^c, A^r; omega), 
also provide its gradients with respect to A^c and to omega
Here we assume that Q is linear in omega: Q(S, A^d, A^c, A^r; omega)=S omega_S+A^d omega_{A^d}+A^c omega_{A^c}+A^r omega_{A^r}
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
        #"Off"=0, "Brk"=-2, "Idl"=-1, "Blk"=1, "Opr"=2#
        #"H"=-1, "K"=0, "W"=1#
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
                list[1-1][i]=0
            elif self.System.machine_states[i]=="Brk":
                list[1-1][i]=-2
            elif self.System.machine_states[i]=="Idl":
                list[1-1][i]=-1
            elif self.System.machine_states[i]=="Blo":
                list[1-1][i]=1
            else:
                list[1-1][i]=2
        for i in range(number_machines-1):
            list[2-1][i]=self.System.buffer_states[i]
        for i in range(3):
            list[3-1][i]=self.System.grid.workingstatus[i]
        for i in range(number_machines):
            if self.System.machine_control_actions[i]=="H":
                list[4-1][i]=-1
            elif self.System.machine_control_actions[i]=="K":
                list[4-1][i]=0
            else:
                list[4-1][i]=1
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
        #calculate the Q-value#
        Q=0
        for i in range(8):
            Q=Q+np.dot(np.array(self.omega[i]), np.array(num_list_States_Actions[i]))    
        return Q
    
    def Q_grad_A_c(self):
        #calculate the gradient of Q with respect to A^c#
        grad=[]
        for i in range(9):
            grad.append(self.omega[6-1][i])
        return np.array(grad)

    def Q_grad_omega(self, num_list_States_Actions):
        #calculate the gradient of Q with respect to omega#
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
        policygradient=np.dot(Q_grad_A_c, A_c_grad_theta)
        return policygradient

    def update(self, policygradient, lr_theta):
        #deterministic policy gradient on theta#
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
    
    #randomly generate an initial theta and plot the bounday of the simplex where theta moves#
    r=np.random.uniform(0,1,size=6)
    theta=[r[0]*r[1], r[0]*(1-r[1]), r[2]*r[3], r[2]*(1-r[3]), r[4]*r[5], r[4]*(1-r[5])] 
    x = [[0, 0], [0, 1], [1, 0]] 
    y = [[0, 1], [1, 0], [0, 0]]
    for i in range(len(x)): 
        plt.plot(x[i], y[i], color='g')
    
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
        #calculate the total cost at S_t, A_t: E(S_t, A_t)#
        E=System.average_total_cost()
        print(" ", file=output)
        print("Average Total Cost=", E, file=output)
        #calculate the old Q-value and its gradient wrt omega: Q(S_t, A_t; omega_t) and grad_omega Q(S_t, A_t; omega_t)#
        av=action_value(System, omega)
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
        print("TD=", TD, file=output)
        #update omega using actor-critique#
        print("Q_grad_omega=", Q_grad_omega_old, file=output)
        factor=lr_omega*TD
        print("lr_omega*TD=", factor, file=output)
        print("omega=", np.array(omega), file=output)
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
        
        #discount the learning rate#
        lr_theta=lr_theta*1
        lr_omega=lr_omega*0.99
        
        print(" ", file=output)
        
        #plot the theta values#
        plt.scatter(theta[0], theta[1], color='b')
        plt.scatter(theta[2], theta[3], marker='+', color='m')
        plt.scatter(theta[4], theta[5], marker='*', color='r')

    plt.show()   
    
    #plot the Q values#
    plt.plot(Q)
    plt.show()   
    
output.close() 
