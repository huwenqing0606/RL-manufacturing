# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:33:36 2020
Modified on Fri, May  5, 15:34:52 2020
@author: Wenqing Hu and Louis Steinmeister
Title: Reinforcement Learning for the joint control of onsite microgrid and manufacturing system
"""

from microgrid_manufacturing_system import Microgrid, ManufacturingSystem, ActionSimulation, MicrogridActionSet_Discrete_Remainder, MachineActionTree, SystemInitialize
from projectionSimplex import projection
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
#import tensorflow.keras.backend as K

#set the number of machines
number_machines=5
#set the unit reward of production
unit_reward_production=1000/10000
#the unit reward for each unit of production (10^4$/unit produced), i.e. the r^p, this applies to the end of the machine sequence#

#the discount factor gamma when calculating the total cost#
gamma=0.999

#the seed for reinforcement training initialization of the network weights and biases
seed=2

#the probability of using random actions vs. on-policy optimal actions in each step of training
p_choose_random_action=0.9

import pandas as pd
#read the solar irradiance, wind speed and the rate of consumption charge data from file#
file_SolarIrradiance = "SolarIrradiance.csv"
file_WindSpeed = "WindSpeed.csv"
file_rateConsumptionCharge = "rate_consumption_charge.csv"
#read the solar irradiace
data_solar = pd.read_csv(file_SolarIrradiance)
solarirradiance = np.array(data_solar.iloc[:,3])
#solar irradiance measured by MegaWatt/km^2
#read the windspeed    
data_wind = pd.read_csv(file_WindSpeed)
windspeed = np.array(data_wind.iloc[:,3])*3.6
#windspeed measured by km/h=1/3.6 m/s
#read the rate of consumption charge
data_rate_consumption_charge = pd.read_csv(file_rateConsumptionCharge)
rate_consumption_charge = np.array(data_rate_consumption_charge.iloc[:,4])/10
#rate of consumption charge measured by 10^4$/MegaWatt=10 $/kWh



"""
Provide the structure of the action-value function Q(S, A^d, A^c, A^r; omega), 
also provide its gradients with respect to A^c and to omega
Here we assume that Q is a 2-hidden layer neural netwok with parameters omega, 
this structure is written into class critic
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
                critic = None
                ):
        #define neural network with 2 hidden layers for the Q function
        self.critic = critic
        self.System=System
        
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
                [0] ]
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
        list[8-1][0]=self.System.grid.actions_discharged #needs to be a list for later convenience
        return list
        
    
    def Q(self, num_list_States_Actions):
        flat_inputs = np.array([item for sublist in num_list_States_Actions for item in sublist],dtype = "float32")
        q = self.critic(flat_inputs)[0,0]
        #print("Evaluation of Critic:", q)
        return q
    
    #has to be called after Q
    def Q_grad_A_c(self):
        #print("DEBUG DQ_Dinput",self.critic.__Q_grad_input__)
        #print("DEBUG DQ_DAc",self.critic.__Q_grad_A_c__)
        return self.critic.__Q_grad_A_c__.numpy()


    def Q_grad_omega(self, num_list_States_Actions):
        #print("Q_grad_omega:", [tensor for tensor in self.critic.__Q_grad_omega__])
        return [tensor.numpy() for tensor in self.critic.__Q_grad_omega__]

    
    def update_weights(self, factor):
        self.critic.update_weights(factor)
        



"""
implements everything related to the Q function
"""
class critic():
    #define the network architecture
    def __init__(self):
        self.run_eagerly = True
        self.dim_input = 3*number_machines-1+18
        print("inputs expected:",self.dim_input)
        #self.input = self.layer_input
        #hidden layers
        self.layer1     = tf.keras.layers.Dense(100, activation='sigmoid', input_shape = (1,self.dim_input))
        self.layer2     = tf.keras.layers.Dense(100, activation='relu')
        #output layer
        self.layer_out  = tf.keras.layers.Dense(1, activation='linear')
        #variable to make sure that gradients are computed before applying
        self.ready_to_apply_gradients = False
        #store the trainable variables
        self.trainable_variables = None
        #execute eager
        

    #evalueate the Q function for a given (state,action)-pair     
    def __call__(self, inputs):
        #flatten inputs for NN

        #the Q value generated by the NN
        #inputs = self.layer_input(flat_inputs)
        input_tensor = tf.reshape(inputs,shape = (1,self.dim_input))
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_tensor)
            #print("DEBUG:",input_tensor)
            res     = self.layer1(input_tensor)
            res     = self.layer2(res)
            Q_value = self.layer_out(res)
        #print("",Q_value)
        if self.trainable_variables == None:
            self.trainable_variables = self.layer1.variables + self.layer2.variables + self.layer_out.variables
            #print("DEBUG weights:",self.trainable_variables)
        #regularization not needed at this time
        #regularization_loss = tf.math.add_n(self.model.losses)         
        #compute the gradients of Q wrt omega
        
        Q_grad_omega = tape.gradient(Q_value, self.trainable_variables)
        #print("Q_grad_omega",Q_grad_omega)
        Q_grad_input = tape.gradient(Q_value, input_tensor)[0]
        #print("Q_grad_input", Q_grad_input)
        Q_grad_A_c   = Q_grad_input[-12:-3] #there are 3 entries after A_c and A_c is 9 long
        self.__Q_grad_omega__ = Q_grad_omega
        self.__Q_grad_input__ = Q_grad_input
        self.__Q_grad_A_c__   = Q_grad_A_c
        self.__Q__            = Q_value
        self.ready_to_apply_gradients = True
        
        return Q_value

    #weight update via gradient descent
    def update_weights(self, factor):
        if self.ready_to_apply_gradients == False: raise Exception("not ready to train the critic. Compute gradients first!")
        #form (gradient, variable)-tuples
        grad_var_tuples = zip(self.__Q_grad_omega__, self.trainable_variables)
        #apply update
        for grad, var in grad_var_tuples:
            #print("TEST")
            #print((var-factor*grad).numpy()[0,0])
            test = (var-factor*grad).numpy()
            while test.ndim>0:
                test = test[0]
            if(math.isnan(test)):
                raise Exception("NaN Produced! Exiting training.")
            else:
                var.assign_add(-factor*grad)
        self.ready_to_apply_gradients = False





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
        print("Policy gradient; Q_grad_A_c:",Q_grad_A_c)
        policygradient=np.dot(Q_grad_A_c, A_c_grad_theta)
        return policygradient

    def update(self, policygradient, lr_theta):
        #deterministic policy gradient on theta#
        theta_old=self.theta
        theta_new=projection(theta_old-lr_theta*policygradient)                           
        return theta_new



"""
Given the current theta and omega, determine the next continuous and discrete/remainder actions by finding 
(1) A^c(theta)=energy distributed in [solar, wind, generator]
(2) with probability probability_randomaction: A^{*,d}_{t+1}=randomly sampled action
    with probability 1-probability_randomaction: A^{*,d}_{t+1}=argmin_{A^d}Q(S_{t+1}, A^d, A^c(theta), A^r(A^d, A^c(theta)); omega)
(3) A^{*,r}_{t+1}=A^r(A^{*,d}_{t+1}, A^c(theta))
"""
def NextAction_OnPolicySimulation(next_machine_states, next_buffer_states, next_workingstatus, next_SOC, t, my_critic, theta, probability_randomaction):
    #build an auxiliarysystem with next system and grid states#
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
    #under the current theta, calculate the continuous actions A^c(theta)=energy distributed in [solar, wind, generator]#
    next_actions_solar=[energy_generated_solar*theta[1-1], energy_generated_solar*theta[2-1], energy_generated_solar*(1-theta[1-1]-theta[2-1])]
    next_actions_wind=[energy_generated_wind*theta[3-1], energy_generated_wind*theta[4-1], energy_generated_wind*(1-theta[3-1]-theta[4-1])]
    next_actions_generator=[energy_generated_generator*theta[5-1], energy_generated_generator*theta[6-1], energy_generated_generator*(1-theta[5-1]-theta[6-1])]
    #tossing_probability is the probability of using randomly simulated actions, and 1-tossing_probability using on-policy actions
    indicator=np.random.binomial(n=1, p=probability_randomaction, size=1)
    if indicator==0:
        #use on-policy actions
        #bulid the list of the set of all admissible machine actions#    
        machine_action_tree=MachineActionTree(machine_action="ROOT")
        machine_action_tree.BuildTree(AuxiliarySystem, level=0, tree=machine_action_tree)
        machine_action_list=[]
        machine_action_tree.TraverseTree(level=0, tree=machine_action_tree, machine_action_list=[])
        machine_action_set_list=machine_action_tree.machine_action_set_list
        #build the list of the set of all admissible microgrid actions for adjusting the status and for purchase/discharge
        microgrid_action_set_DR=MicrogridActionSet_Discrete_Remainder(AuxiliarySystem)
        microgrid_action_set_list_adjustingstatus=microgrid_action_set_DR.List_AdjustingStatus()
        microgrid_action_set_list_purchased_discharged=microgrid_action_set_DR.List_PurchasedDischarged(actions_solar=next_actions_solar,
                                                                                                        actions_wind=next_actions_wind,
                                                                                                        actions_generator=next_actions_generator)
        optimal_Q=0
        next_machine_actions=[]
        next_microgrid_actions_adjustingstatus=[]
        next_microgrid_actions_purchased=[]
        next_microgrid_actions_discharged=0
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
                                                                       actions_solar=next_actions_solar,
                                                                       actions_wind=next_actions_wind,
                                                                       actions_generator=next_actions_generator,
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
                        next_machine_actions=machine_action_list
                        next_microgrid_actions_adjustingstatus=microgrid_action_list_adjustingstatus
                        next_microgrid_actions_purchased=microgrid_action_list_purchased_discharged[0]
                        next_microgrid_actions_discharged=microgrid_action_list_purchased_discharged[1]
                    else:
                        if Q<optimal_Q:
                            optimal_Q=Q
                            next_machine_actions=machine_action_list
                            next_microgrid_actions_adjustingstatus=microgrid_action_list_adjustingstatus
                            next_microgrid_actions_purchased=microgrid_action_list_purchased_discharged[0]
                            next_microgrid_actions_discharged=microgrid_action_list_purchased_discharged[1]
                    i=i+1
    else:
        #use randomly sampled actions
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
        next_microgrid_actions_adjustingstatus=next_action.MicroGridActions_adjustingstatus()
        next_actions_solar, next_actions_wind, next_actions_generator=next_action.MicroGridActions_SolarWindGenerator(theta)
        next_microgrid_actions_purchased, next_microgrid_actions_discharged=next_action.MicroGridActions_PurchasedDischarged(next_actions_solar,
                                                                                                                             next_actions_wind,
                                                                                                                             next_actions_generator)
        next_machine_actions=next_action.MachineActions()
                    
                
                
    return next_machine_actions, next_actions_solar, next_actions_wind, next_actions_generator, next_microgrid_actions_adjustingstatus, next_microgrid_actions_purchased, next_microgrid_actions_discharged




"""
Reinforcement Learning Algorithm: On-policy TD control combined with actor-critique
Algorithm 1 in the paper
The Training Process of the Reinforcement Learning Algorithm
"""
def Reinforcement_Learning_Training(System_init, #the initial system
                                    theta_init,  #the initial theta value
                                    lr_theta_init, #the initial learning rate in the theta iteration
                                    lr_omega_init, #the initial learning rate in the omega iteration
                                    number_iteration, #the total number of training iterations
                                    ):
    
    tf.enable_eager_execution()
    #set seed for tensorflow (including initialization of weights and bias) for reproducability
    tf.set_random_seed(seed)     
    #K.clear_session()

    #initialize#
    theta=theta_init
    System=System_init
    lr_tht=lr_theta_init
    lr_omg=lr_omega_init
    
    #the critic is with parameters given by omega
    my_critic = critic()
    omega = []
    
    #initialize the sequence of Q values, Temporal Difference and the reward, the reward sequence and L^2 difference in omega
    Q=[]
    TemporalDifference=[]
    rewardseq=[]
    reward=0
    diff_omega=[]
    
    ################### reinforcement learning training process ###################
    for t in range(number_iteration):
        print("---------- iteration", t, "----------")
        #calculate the total cost at S_t, A_t: E(S_t, A_t)
        E=System.average_total_cost(rate_consumption_charge[t//8640])
        #calculate the old Q-value and its gradient wrt omega: Q(S_t, A_t; omega_t) and grad_omega Q(S_t, A_t; omega_t)
        av=action_value(System, my_critic)
        num_list_SA=av.num_list_States_Actions()
        Q_old=av.Q(num_list_SA)
        Q.append(Q_old)
        #update the theta using deterministic policy gradient#
        upd_tht=update_theta(System, theta)
        policy_grad=upd_tht.deterministic_policygradient(upd_tht.A_c_gradient_theta(), av.Q_grad_A_c())
        theta=upd_tht.update(policy_grad, lr_tht)
        #calculate the next states and actions: S_{t+1}, A_{t+1}#        
        next_machine_states, next_buffer_states=System.transition_manufacturing()
        next_workingstatus, next_SOC=System.grid.transition()
        #determine A^c(theta)=energy distributed in [solar, wind, generator]
        #determine A^{*,d}_{t+1}=argmin_{A^d}Q(S_{t+1}, A^d, A^c(theta), A^r(A^d, A^c(theta)); omega)
        #determine A^{*,r}_{t+1}=A^r(A^{*,d}_{t+1}, A^c(theta))
        next_machine_actions, next_actions_solar, next_actions_wind, next_actions_generator, next_microgrid_actions_adjustingstatus, next_microgrid_actions_purchased, next_microgrid_actions_discharged = NextAction_OnPolicySimulation(next_machine_states, 
                                                                                                                                                                                                                                         next_buffer_states, 
                                                                                                                                                                                                                                         next_workingstatus, 
                                                                                                                                                                                                                                         next_SOC,
                                                                                                                                                                                                                                         t, 
                                                                                                                                                                                                                                         my_critic,
                                                                                                                                                                                                                                         theta,
                                                                                                                                                                                                                                         probability_randomaction=p_choose_random_action)
        grid=Microgrid(workingstatus=next_workingstatus,
                       SOC=next_SOC,
                       actions_adjustingstatus=next_microgrid_actions_adjustingstatus,
                       actions_solar=next_actions_solar,
                       actions_wind=next_actions_wind,
                       actions_generator=next_actions_generator,
                       actions_purchased=next_microgrid_actions_purchased,
                       actions_discharged=next_microgrid_actions_discharged,
                       solarirradiance=solarirradiance[t//8640],
                       windspeed=windspeed[t//8640]
                       )
        System=ManufacturingSystem(machine_states=next_machine_states, 
                                   machine_control_actions=next_machine_actions, 
                                   buffer_states=next_buffer_states,
                                   grid=grid
                                   )        
        
        #TD-control SARSA#
        av=action_value(System, my_critic)
        num_list_SA=av.num_list_States_Actions()
        Q_new=av.Q(num_list_SA)
        TD=E+gamma*Q_new-Q_old
        TemporalDifference.append(TD)
        #calculate the up-to-date reward#
        reward=reward+np.power(gamma, t)*E
        rewardseq.append(reward)
        #update omega using actor-critique#
        factor=lr_omg*TD
        my_critic.update_weights(factor)
        #update and calculate the norm difference in omega#
        if t==0:
            diff_omega.append(0)
            omega = [var.numpy() for var in my_critic.trainable_variables]
            omega_init = omega
        else:
            omega_old=omega
            omega = [var.numpy() for var in my_critic.trainable_variables]
            omega_new=omega
            diff_omega.append(np.sum([np.linalg.norm(new_var-old_var) for new_var, old_var in zip(omega_new, omega_old)]))
       
        #discount the learning rate#
        lr_tht=lr_tht*1
        lr_omg=lr_omg*0.999
                
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
    plt.plot([value*10000 for value in rewardseq])
    plt.xlabel('iteration')
    plt.ylabel('Sum of rewards during episode ($)')
    plt.savefig('rewards.png')
    plt.show()       

    #plot the weight difference sequences#
    plt.figure(figsize = (14,10))
    plt.plot(diff_omega)
    plt.xlabel('iteration')
    plt.ylabel('L2 norm of the difference in the weights')
    plt.savefig('weightdifference.png')
    plt.show()   
    
    #record the training process onto a file train_output.txt    
    troutput = open('train_output.txt', 'w')
    print("************************ Reinforcement Learning Training at "+str(number_iteration)+" steps ************************", file=troutput)

    #print the initial theta and optimal theta after training
    print("\nthe initial theta is given by: ", theta_init, file=troutput)
    print("\nthe optimal theta after training is given by: ", theta, file=troutput)
    #print the L2 difference of the theta between initial and final
    print("\nthe theta moves at a L^2 distance: ", np.sum([np.linalg.norm(new_var-old_var) for new_var, old_var in zip(theta, theta_init)]), file=troutput)

    #print the initial omega and optimal omega after training for the critic parameters
    print("\nthe seed for choosing initial weight and bias parameter is given by: ", seed, file=troutput)
    print("\nthe initial omega for the critic is given by: ", omega_init, file=troutput)
    print("\nthe optimal omega for the critic after training is given by: ", omega, file=troutput)
    #print the L2 difference of the omega between initial and final
    print("\nthe omega moves at a L^2 distance: ", np.sum([np.linalg.norm(new_var-old_var) for new_var, old_var in zip(omega, omega_init)]), file=troutput)
        
    #close and save the training ouput file
    troutput.close() 
    
    return theta, omega, my_critic
    




"""
Reinforcement Learning Algorithm: Off policy TD control combined with actor-critique
Algorithm 1 in the paper
The Testing Process of the Reinforcement Learning Algorithm
Output at a given horizon the system dynamics under the optimal action selected by Reinforcement Learning training
select the best action-value function at each iteration
"""
def Reinforcement_Learning_Testing(System_init, #the initial point of the system dynamics
                                   thetainit,   #the initial theta before training
                                   thetaoptimal, #the optimal theta obtained after RL training
                                   omegaoptimal,    #the optimal parameter of critic obtained after training
                                   my_critic_optimal, #the optimal critic obtained after training
                                   number_iteration, #the total number of testing iterations
                                   unit_reward_production #the unit reward of production used to calculate target output
                                   ):
    
    #output the result to a file test_output.txt#
    testoutput = open('test_output.txt', 'w') 

    print("************************* Optimal System with optimal policy *************************", file=testoutput)
    print("***Run the system on optimal policy at a time horizon=", number_iteration,"***", file=testoutput)
    print("\n", file=testoutput)
    print("initial proportion of energy supply=", thetainit, file=testoutput)
    print("optimal proportion of energy supply=", thetaoptimal, file=testoutput)
    print("optimal parameter for the neural-network integrated action-value function=", omegaoptimal, file=testoutput)
    print("\n", file=testoutput)
    #run the MDP under optimal theta and optimal omega#
    #at every step search among all discrete actions to find A^d_*=argmin_{A^d}Q(A^d, A^c(thetaoptimal), A^r(A^d, A^c(thetaoptimal)))#
    #Calculate 1. Total cost (E) and throughput in given time horizon that the algorithm is used to guide the bilateral control#
    #Calculate 2. Total energy demand across all time periods of the given time horizon#

    #set the initial point of running the system
    System=System_init
    #initialize the list of total cost, total throughput and total energy demand that will be returned
    totalcostlist_optimal=[0]
    totalthroughputlist_optimal=[0]
    totalenergydemandlist_optimal=[0]
    
    #set the total cost, total throughput and the total energy demand#
    totalcost=0
    totalthroughput=0
    totalenergydemand=0 
    RL_target_output=0
    #reinforcement learning testing loop
    for t in range(number_iteration):
        #start of the iteration loop for reinforcement learning testing process#
        #current states and actions S_t and A_t are stored in class System#
        #Print (S_t, A_t)#
        print("*********************Time Step", t, "*********************", file=testoutput)
        System.PrintSystem(testoutput, t)
        #accumulate the total throughput#
        totalthroughput+=System.throughput()
        RL_target_output+=int(System.throughput()/unit_reward_production)
        totalthroughputlist_optimal.append(totalthroughput)
        #calculate the total cost at S_t, A_t: E(S_t, A_t)#
        E=System.average_total_cost(rate_consumption_charge[t//8640])
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
        #determine the next discrete actions by finding A^{*,d}_{t+1}=argmin_{A^d}Q(S_{t+1}, A^d, A^c(thetaoptimal), A^r(A^d, A^c(thetaoptimal)); omegaoptimal)#
        #determine the next remainder actions A^{*,r}_{t+1}=A^r(A^{*,d}_{t+1}, A^c(thetaoptimal))
        optimal_next_machine_actions, optimal_next_actions_solar, optimal_next_actions_wind, optimal_next_actions_generator, optimal_next_microgrid_actions_adjustingstatus, optimal_next_microgrid_actions_purchased, optimal_next_microgrid_actions_discharged = NextAction_OnPolicySimulation(next_machine_states, 
                                                                                                                                                                                                                                                                                                 next_buffer_states, 
                                                                                                                                                                                                                                                                                                 next_workingstatus, 
                                                                                                                                                                                                                                                                                                 next_SOC,
                                                                                                                                                                                                                                                                                                 t, 
                                                                                                                                                                                                                                                                                                 my_critic_optimal,
                                                                                                                                                                                                                                                                                                 thetaoptimal,
                                                                                                                                                                                                                                                                                                 probability_randomaction=0)
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

    print("\n****************** SUMMARY *******************", file=testoutput)
    print("\ntotal cost list (10^4$) =", totalcostlist_optimal, file=testoutput)
    print("\ntotal throughput list (10^4$) =", totalthroughputlist_optimal, file=testoutput)
    print("\ntotal energy demand list (10^4$) =", totalenergydemandlist_optimal, file=testoutput)

    print("\ntotal cost ($) =", totalcost*10000, file=testoutput)    
    print("\ntotal throughput ($) =", totalthroughput*10000, file=testoutput)    
    print("\ntotal energy demand ($) =", totalenergydemand*10000, file=testoutput)
    print("\ntarget output (unit) =", RL_target_output, file=testoutput)
    
    #close and save the test output file
    testoutput.close()

    return totalcostlist_optimal, totalthroughputlist_optimal, totalenergydemandlist_optimal, RL_target_output





"""
Benchmark Testing Process of the Manufacturing System Dynamics under randomly selected actions
Output at a given horizon the system dynamics under randomly selected actions
"""
def Benchmark_RandomAction_Testing(System_init, #the inital point of running the system dynamics
                                   thetainit, #the initial theta used before RL training
                                   number_iteration, #the number of training iterations
                                   unit_reward_production #the unit reward of production used to calculate target output
                                   ):

    #output the result to a file benchmark_output.txt
    bmoutput = open('benchmark_output.txt', 'w')

    #As benchmark, with initial theta and randomly simulated actions, run the system at a certain time horizon#
    print("************************* BenchMark System with initial theta and random actions *************************", file=bmoutput)
    print("***Run the system on random policy at a time horizon=", number_iteration,"***", file=bmoutput)
    print("\n", file=bmoutput)
    print("initial proportion of energy supply=", thetainit, file=bmoutput)
    print("\n", file=bmoutput)

    #set the initial point of running the system
    System=System_init

    #compare the optimal control and random control (benchmark)#
    totalcostlist_benchmark=[0]
    totalthroughputlist_benchmark=[0]
    totalenergydemandlist_benchmark=[0]

    #set the total cost, total throughput and the total energy demand#
    totalcost=0
    totalthroughput=0
    totalenergydemand=0 
    random_target_output=0
    #benchmark system iteration loop
    for t in range(number_iteration):
        #start of the iteration loop for a benchmark system with initial theta and random actions#
        #current states and actions S_t and A_t are stored in class System#
        #Print (S_t, A_t)#
        print("*********************Time Step", t, "*********************", file=bmoutput)
        System.PrintSystem(bmoutput, t)
        #accumulate the total throughput#
        totalthroughput+=System.throughput()
        random_target_output+=int(System.throughput()/unit_reward_production)
        totalthroughputlist_benchmark.append(totalthroughput)
        #calculate the total cost at S_t, A_t: E(S_t, A_t)#
        E=System.average_total_cost(rate_consumption_charge[t//8640])
        #accumulate the total cost#
        totalcost+=E
        totalcostlist_benchmark.append(totalcost)
        #accumulate the total energy demand#
        totalenergydemand+=System.energydemand(rate_consumption_charge[t//8640])
        totalenergydemandlist_benchmark.append(totalenergydemand)
        #determine the next system and grid states#
        next_machine_states, next_buffer_states=System.transition_manufacturing()
        next_workingstatus, next_SOC=System.grid.transition()
        #update the manufacturing system and the grid according to S_{t+1}, A^{d}_{t+1}, A^c(thetainit), A^{r}_{t+1}#
        my_critic = critic()
        next_machine_actions, next_actions_solar, next_actions_wind, next_actions_generator, next_microgrid_actions_adjustingstatus, next_microgrid_actions_purchased, next_microgrid_actions_discharged = NextAction_OnPolicySimulation(next_machine_states, 
                                                                                                                                                                                                                                         next_buffer_states, 
                                                                                                                                                                                                                                         next_workingstatus, 
                                                                                                                                                                                                                                         next_SOC,
                                                                                                                                                                                                                                         t, 
                                                                                                                                                                                                                                         my_critic,
                                                                                                                                                                                                                                         theta=thetainit,
                                                                                                                                                                                                                                         probability_randomaction=1)
        
        grid=Microgrid(workingstatus=next_workingstatus,
                       SOC=next_SOC,
                       actions_adjustingstatus=next_microgrid_actions_adjustingstatus,
                       actions_solar=next_actions_solar,
                       actions_wind=next_actions_wind,
                       actions_generator=next_actions_generator,
                       actions_purchased=next_microgrid_actions_purchased,
                       actions_discharged=next_microgrid_actions_discharged,
                       solarirradiance=solarirradiance[t//8640],
                       windspeed=windspeed[t//8640]
                       )
        System=ManufacturingSystem(machine_states=next_machine_states, 
                                   machine_control_actions=next_machine_actions, 
                                   buffer_states=next_buffer_states,
                                   grid=grid
                                   )        
        #end of the iteration loop for for a benchmark system with initial theta and random actions#

    print("\n****************** SUMMARY *******************", file=bmoutput)    
    print("\ntotal cost list (10^4$) =", totalcostlist_benchmark, file=bmoutput)
    print("\ntotal throughput list (10^4$) =", totalthroughputlist_benchmark, file=bmoutput)
    print("\ntotal energy demand list (10^4$) =", totalenergydemandlist_benchmark, file=bmoutput)
    
    print("\ntotal cost ($) =", totalcost*10000, file=bmoutput)    
    print("\ntotal throughput ($) =", totalthroughput*10000, file=bmoutput)   
    print("\ntotal energy demand ($) =", totalenergydemand*10000, file=bmoutput)
    print("\ntarget output (unit) =", random_target_output, file=bmoutput)
    
    #close and save the benchmark output file
    bmoutput.close()
    
    return totalcostlist_benchmark, totalthroughputlist_benchmark, totalenergydemandlist_benchmark, random_target_output




"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

Testing the Reinforcement Learning Algorithm: On-policy TD control combined with actor-critique
Algorithm 1 in the paper

Compare its behavior with randomly selected actions

When optimal policy is found, must add
1. Total cost and throughput in given time horizon that the 
   algorithm is used to guide the bilateral control.
2. Total energy demand across all time periods of the given 
   time horizon and the proportion of the energy supply to satisfy the demand. 
"""

if __name__ == "__main__":
    #the initial learning rates for the theta and omega iterations#
    lr_theta_initial=0.003
    lr_omega_initial=0.0003

    #number of training and testing iterations#
    training_number_iteration=5
    testing_number_iteration=100


    #set the initial machine states, machine control actions and buffer states
    initial_machine_states=["Opr" for _ in range(number_machines)]
    initial_machine_actions=["K" for _ in range(number_machines)]
    initial_buffer_states=[2 for _ in range(number_machines-1)]
    
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
    
    
    theta, omega, my_critic = Reinforcement_Learning_Training(System, 
                                                              thetainit, 
                                                              lr_theta_initial, 
                                                              lr_omega_initial, 
                                                              training_number_iteration
                                                              )
    
    
    #with the optimal theta and optimal omega at hand, run the system at a certain time horizon#
    #output the optimal theta and optimal omega#
    thetaoptimal=theta
    omegaoptimal=omega  
    my_critic_optimal=my_critic

    #initialize the system
    System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)

    totalcostlist_optimal, totalthroughputlist_optimal, totalenergydemandlist_optimal, RL_target_output = Reinforcement_Learning_Testing(System, 
                                                                                                                                         thetainit, 
                                                                                                                                         thetaoptimal, 
                                                                                                                                         omegaoptimal, 
                                                                                                                                         my_critic_optimal, 
                                                                                                                                         testing_number_iteration, 
                                                                                                                                         unit_reward_production)
    
    #As benchmark, with initial theta and randomly simulated actions, run the system at a certain time horizon#
    
    #initialize the system
    System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)

    totalcostlist_benchmark, totalthroughputlist_benchmark, totalenergydemandlist_benchmark, random_target_output = Benchmark_RandomAction_Testing(System, 
                                                                                                                                                   thetainit, 
                                                                                                                                                   testing_number_iteration, 
                                                                                                                                                   unit_reward_production 
                                                                                                                                                   )

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