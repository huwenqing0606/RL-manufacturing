# -*- coding: utf-8 -*-
"""
code for the manufactoring system
Author: Yunzhao Zhang and Wenqing Hu

"""

import numpy as np

class Machine(object):
    debugMode = True
    def __init__(self, 
                 fail_rate=0.0333, 
                 mttr=10, 
                 prod_rate=1, 
                 input_rate=1, 
                 pow_on=1000, 
                 pow_idle=600, 
                 pow_off=0, 
                 switch=0, 
                 states="OFF", 
                 t_repair=0, 
                 source=None, 
                 output=None, name = "Machine"):
        """
        fail_rate: probability that machine fails
        mttr: mean time for the machine to be repared
        prod_rate: production rate at each time interval
        pow_on: power consumption when state is ON 
        pow_idle: power consumption when state is IDLE
        pow_off: power consumtion when state is OFF
        states: machine states (ON, IDLE, OFF)
        source: from where it takes input, if none then infinite resource 
        output: to where it outputs products, if none then infinite storage capacity
        """
#       print("A machine is built!")
        self.fail_rate = fail_rate
        self.mttr = mttr
        self.prod_rate = prod_rate
        self.pow_on = pow_on
        self.pow_idle = pow_idle 
        self.pow_off = pow_off 
        self.switch = switch
        self.states = states
        self.source = source
        self.output = output
        self.t_repair = t_repair
        self.input_rate = input_rate
        self.name = name
    
    def set_debug(self, debugMode=True):
        self.debugMode = debugMode
        
    def on_action(self, switch):
        """
        switch: turn the machine on or off
        """
        self.switch = switch
        if self.debugMode:
            print()
            print("Machine",self.name, "switch status: ", self.switch)
            print()
    def on_cycle(self, cycle_time = 2):
        """
        cycle_time: time takes to run one cycle 
        """
        source = self.source
        output = self.output
        consumed = 0
        produced = 0
        # check for switch status, if off then state is OFF, else check for buffer
        if self.switch == 0 or self.t_repair>0:
            self.states = "OFF"
        else:
            if source == None and output == None:
                self.states = "ON"
            elif source and output:
                if source.hasEnough(self.input_rate) and not output.willFull(self.prod_rate):
                    self.states = "ON"
                else:
#                     print("case 1")
                    self.states = "IDLE"
            elif source and output == None:
                if source.hasEnough(self.input_rate):
                    self.states = "ON"
                else:
#                     print("case 2")
                    self.states = "IDLE"
            elif source == None and output:
                if not output.willFull(self.prod_rate):
                    self.states = "ON"
                else:
#                     print("case 3")
                    self.states = "IDLE"
                    
        # check if it is being repaired
        if self.t_repair > 0:
            self.t_repair -= cycle_time
        else:
            # determine if the machine will fail 
            randn = np.random.uniform(0,1)
            if randn < self.fail_rate:
                self.t_repair = int((self.mttr/4)*np.random.randn()+self.mttr)
                self.states = "OFF"
                if self.debugMode:
                    print("OH NO!!! The machine fails... Need repair time: ", self.t_repair)
            elif self.states == "ON":
                if source:
                    consumed = source.consume(self.input_rate)
                if output:
                    produced = output.store(self.prod_rate)
                
                

        if self.debugMode:
            self.print_status(produced=produced, consumed=consumed)
            
    def print_status(self, produced=0, consumed=0):
        print()
        print("Machine",self.name, " is currently", self.states)            
        print("Remaining repairing time: ", max(0,self.t_repair))
        print("Resource consumed: ", consumed)
        print("Product produced: ", produced)
        print()

class Buffer(object):
    def __init__(self, initial_amt = 0, capacity = 20):
        self.amount = initial_amt
        self.capacity = capacity
    def hasEnough(self, taken):
        if self.capacity == -1:
            return True
        return self.amount >= taken
    def willFull(self, gotten):
        if self.capacity == -1:
            return False
        return self.amount+gotten > self.capacity
    def consume(self, taken):
        self.amount -= taken
        return taken
    def store(self, gotten):
        self.amount += gotten
        return gotten

import pandas as pd

class ManufacturingSystem(object):
    def __init__(self, cycle_time = 2, num_machines = 2, state_time = 60, solar_irr=None, wind_sp=None, 
                 battery_cap=800, generator_cap=400, period = 8640, e_c=0.9, e_dc=0.9, debugMode=True):
        self.time = 0
        self.cycle_time = cycle_time
        self.num_machines = num_machines
        self.buf0 = Buffer(capacity=-1)
        self.buf1 = Buffer(initial_amt=10, capacity=20)
        self.buf2 = Buffer(capacity=-1)
        self.mac0 = Machine(source=self.buf0, output=self.buf1, name="Machine 1")
        self.mac1 = Machine(source=self.buf1, output=self.buf2, name="Machine 2")
        self.machines = [self.mac0, self.mac1]
        self.buffers = [self.buf1, self.buf2]
        self.state_time = state_time
        self.solar_irr = None
        self.wind_sp = None
        self.battery_cap = battery_cap
        self.generator_cap = generator_cap
        self.period = period
        self.e_c = e_c
        self.e_dc = e_dc
        
        
        self.cost_solar = 0.02
        self.cost_wind = 0.03
        self.cost_battery = 0.1
        self.cost_generator = 0.2
        self.cost_utility = [0.35,0.19,0.06]
        self.price_soldback = [0.17,0.07,0]
        
        
        '''
        Energy components
        ''' 
        # Solar charging state
        self.solar_irr = solar_irr
        self.wind_sp = wind_sp
            
        
        '''
        Initial states
        '''
        self.states = []
        # M_i: states of machines, 3 vars for each (ON, IDLE, OFF)                                              
        for i in range(self.num_machines):
            self.states.append(0)
            self.states.append(0)
            self.states.append(1)
        # B_i: states of intermediate buffers
        for i in range(self.num_machines-1):
            self.states.append(self.buffers[i].amount)
        # Y: total production
        self.states.append(self.buf2.amount)
        # S: current solar energy charging rate
        self.states.append(0)
        # W: current wind energy charging rate 
        self.states.append(0)
        # G: current generator rate
        self.states.append(0)
        # SOC: state of charge of the batter
        self.states.append( 0.6)
        # SB: sold back rate
        self.states.append(0)
        # U: utility purchase
        self.states.append(0)
        # I: current solar irradiance
        self.states.append(self.solar_irr[0])
        # F: current wind speed
        self.states.append(self.wind_sp[0])
        # t: time of period 
        self.states.append(self.time//60)
        
        
        
    
    def on_cycle(self):
        self.time += self.cycle_time
        self.mac0.on_cycle(cycle_time=self.cycle_time)
        self.mac1.on_cycle(cycle_time=self.cycle_time)
        
        energy = 0
        for mac in self.machines:
            if mac.states == "ON":
                energy += mac.pow_on * self.cycle_time / self.state_time
            elif mac.states == "IDLE":
                energy += mac.pow_idle * self.cycle_time / self.state_time
            elif mac.states == "OFF":
                energy += mac.pow_off * self.cycle_time / self.state_time
        return energy
    
    def get_distribution(self, actions):
        demand = 0
        
        for i, mac in enumerate(self.machines):
            mac.on_action(actions[i])        

            
        for t in range(0, self.state_time//self.cycle_time):
#             print("------------------------------- Cycle {} -------------------------------".format(t))
            demand += self.on_cycle()
#             print()
#             self.print_status()
        
#         print("Energy Demand:",demand)
        return demand
        
        
        
        
    def on_action(self, actions):
        '''
        Actions:
        [0.. num_machines]: control actions of the machines
        [num_machines]: solar energy to manufacturing system
        [num_machines+1]: solar charge to battery 
        [num_machines+2]: solar energy sold back 
        [num_machines+3]: wind energy to manufacturing system
        [num_machines+4]: wind charge to battery 
        [num_machines+5]: wind energy to sold back 
        [num_machines+6]: generator energy to manufacturing system
        [num_machines+7]: discharging from battery to manufacturing system
        [num_machines+8]: utility purchased to manufacturing system 
        '''
        demand = 0
        
        for i, mac in enumerate(self.machines):
            mac.on_action(actions[i])
        
        
        for t in range(0, self.state_time//self.cycle_time):
#             print("------------------------------- Cycle {} -------------------------------".format(t))
            demand += self.on_cycle()
#             print()
#             self.print_status()
        
#         print("Energy Demand:",demand)
        # update the states of the manufacturing system
        pre_state = self.states
        self.update_states(actions)
        post_state = self.states 
        acts = self.get_actions()
        energy_cost = self.calc_cost(demand, actions)
        
#         print("Cost: ",energy_cost)
        return post_state, energy_cost, acts
        
    
    def get_actions(self):
        '''
        put after state is updated
        '''
        
        actions = []
        cost_solar = self.cost_solar
        cost_wind = self.cost_wind
        cost_battery = self.cost_battery
        cost_generator = self.cost_generator

        tf = (self.states[-1]) % 24
        if tf > 13 and tf <= 19:
            tf = 0
        elif (tf > 10 and tf <=13) or (tf > 18 and tf <= 21):
            tf = 1
        else:
            tf = 2
        cost_utility = self.cost_utility[tf]
        price_soldback = self.price_soldback[tf]
        
        both_on = np.load("both_on.npy")
        both_off = np.load("both_off.npy")
        st_on = np.load("1st_on.npy")
        nd_on = np.load("2nd_on.npy")
        demands = [both_on, both_off, st_on, nd_on]
        
        
        for mac1 in range(2):
            for mac2 in range(2):
                demand = demands[0][self.states[-1]]
                if mac1 == 0 and mac2 == 0:
                    demand = demands[1][self.states[-1]]
                elif mac1 == 1 and mac2 == 0:
                    demand = demands[2][self.states[-1]]
                elif mac1 == 0 and mac2 == 1:
                    demand = demands[3][self.states[-1]]
                use_wind = 0
                use_solar = 0
                use_battery = 0
                use_generator = 0
                use_utility = 0
                wind_energy = self.wind_sp[self.states[-1]]
                solar_energy = self.solar_irr[self.states[-1]]
                battery_energy = self.states[self.num_machines*3+self.num_machines+3]*self.battery_cap
                generator_energy = self.generator_cap
                
#                 print("Demand: ", demand, "Solar: ", solar_energy, "Wind: ", wind_energy)
                
                # use up cheapest until demand energy is satisfied
                if demand > 0 and cost_solar < cost_utility and cost_solar < cost_wind:
                    if solar_energy > demand:
                        use_solar = demand
                        solar_energy -= demand
                        demand = 0
                    else:
                        use_solar = solar_energy
                        demand -= solar_energy
                        solar_energy = 0
                if demand > 0 and cost_wind < cost_utility and cost_wind < cost_solar:
                    if wind_energy > demand:
                        use_wind = demand
                        wind_energy -= demand 
                        demand = 0
                    else:
                        use_wind = wind_energy
                        demand -= wind_energy
                        wind_energy = 0
                if demand > 0 and cost_solar < cost_utility and solar_energy>0:
                    if solar_energy > demand:
                        use_solar = demand
                        solar_energy -= demand
                        demand = 0
                    else:
                        use_solar = solar_energy
                        demand -= solar_energy
                        solar_energy = 0
                if demand > 0 and cost_wind < cost_utility and wind_energy>0:
                    if wind_energy > demand:
                        use_wind = demand
                        wind_energy -= demand 
                        demand = 0
                    else:
                        use_wind = wind_energy
                        demand -= wind_energy
                        wind_energy = 0
                use_batery = 0
                use_generator = 0
                if demand > 0 and cost_battery < cost_utility and cost_battery < cost_generator:
                    if battery_energy > demand:
                        use_battery += demand
                        battery_energy -= demand 
                        demand = 0
                    else:
                        use_battery += battery_energy
                        demand -= battery_energy
                        battery_energy = 0
                if demand > 0 and cost_generator < cost_utility and cost_generator < cost_battery:
                    if generator_energy > demand:
                        use_generator += demand
                        generator_energy -= demand 
                        demand = 0
                    else:
                        use_generator += generator_energy
                        demand -= generator_energy
                        generator_energy = 0
                        
                if demand > 0 and cost_battery < cost_utility and battery_energy > 0:
                    if battery_energy > demand:
                        use_battery += demand
                        battery_energy -= demand 
                        demand = 0
                    else:
                        use_battery += battery_energy
                        demand -= battery_energy
                        battery_energy = 0      
                        
                if demand > 0 and cost_generator < cost_utility and generator_energy > 0:
                    if generator_energy > demand:
                        use_generator += demand
                        generator_energy -= demand 
                        demand = 0
                    else:
                        use_generator += generator_energy
                        demand -= generator_energy
                        generator_energy = 0
                    
                remain_SOC = (self.battery_cap - battery_energy) / self.battery_cap
                use_utility = demand
                
#                 print("remain_SOC: ", remain_SOC)
                for sb in range(0,int(remain_SOC*100),5):
                    solar_charge = sb*self.battery_cap/100 
                    if solar_charge <= solar_energy:
                        for wb in range(0,int(remain_SOC*100-sb), 5):
                            wind_charge = wb*self.battery_cap/100
                            if wind_charge <= wind_energy:
                                temp_solar = solar_energy - solar_charge
                                temp_wind = wind_energy - wind_charge
                                solar_sold = 0
                                wind_sold = 0
                                if cost_solar < price_soldback: 
                                    solar_sold = temp_solar
                                if cost_wind < price_soldback:
                                    wind_sold = temp_wind
                                action = [mac1, mac2, use_solar, solar_charge, solar_sold, use_wind, wind_charge, wind_sold, use_generator, use_battery, use_utility]    
                                actions.append(action)
                
                
        
        
                
        
        return actions
       
            
    
    def calc_cost(self, demand, actions):
        cost_solar = self.cost_solar
        cost_wind = self.cost_wind
        cost_battery = self.cost_battery
        cost_generator = self.cost_generator
        cost_utility = self.cost_utility
        price_soldback = self.price_soldback
        cost = 0
        
        # time frame (on-peak, mid-peak, off-peak)
        tf = self.states[-1] % 24
        if tf > 13 and tf <= 19:
            tf = 0
        elif (tf > 10 and tf <=13) or (tf > 18 and tf < 21):
            tf = 1
        else:
            tf = 2
        
        # cost of solar energy
        cost += (actions[self.num_machines]+actions[self.num_machines+1]+actions[self.num_machines+2])*cost_solar
        # cost of wind energy
        cost += (actions[self.num_machines+3]+actions[self.num_machines+4]+actions[self.num_machines+5])*cost_wind
        # cost of generator energy
        cost += actions[self.num_machines+6]*cost_generator 
        # cost of battery
        cost += (actions[self.num_machines+1]+actions[self.num_machines+4]+actions[self.num_machines+7])*cost_battery
        # cost of utility
        cost += actions[self.num_machines+8]*cost_utility[tf]
        # earned of sold back
        cost -= (actions[self.num_machines+2]+actions[self.num_machines+5])*price_soldback[tf]
        

        
        return cost

    
        
    def update_states(self, actions):
        # M_i: states of machines, 3 vars for each                                                 
        for i,mac in enumerate(self.machines):
            if mac.states == "ON":
                self.states[i*3] = 1
                self.states[i*3+1] = 0
                self.states[i*3+2] = 0
            elif mac.states == "IDLE":
                self.states[i*3] = 0
                self.states[i*3+1] = 1
                self.states[i*3+2] = 0    
            elif mac.states == "OFF":
                self.states[i*3] = 0
                self.states[i*3+1] = 0
                self.states[i*3+2] = 1 
        # B_i: states of intermediate buffers
        for i in range(self.num_machines-1):
            self.states[self.num_machines*3+i] = self.buffers[i].amount 
        # Y: total production
#         self.states[self.num_machines*3+self.num_machines-1] = self.buffers[-1].amount
        # S: next solar energy charging rate
        self.states[self.num_machines*3+self.num_machines] = actions[self.num_machines]+actions[self.num_machines+1]+actions[self.num_machines+2]
        # W: next wind energy charging rate 
        self.states[self.num_machines*3+self.num_machines+1] = actions[self.num_machines+3]+actions[self.num_machines+4]+actions[self.num_machines+5]
        # G: next generator rate
        self.states[self.num_machines*3+self.num_machines+2] = actions[self.num_machines+6]
        # SOC: state of charge of the battery
        total_charging = actions[self.num_machines+1]+actions[self.num_machines+4]
        self.states[self.num_machines*3+self.num_machines+3] = (self.states[self.num_machines*3+self.num_machines+3]*self.battery_cap+self.e_c*total_charging-actions[self.num_machines+7]/self.e_dc) / self.battery_cap
        # SB: sold back rate
        self.states[self.num_machines*3+self.num_machines+4] = actions[self.num_machines+2]+actions[self.num_machines+5]
        # U: utility purchase
        self.states[self.num_machines*3+self.num_machines+5] = actions[self.num_machines+8]
        # t: time of period 
        self.states[self.num_machines*3+self.num_machines+8] += 1
        self.states[self.num_machines*3+self.num_machines+8] = self.states[self.num_machines*3+self.num_machines+8] % self.period
#         if self.states[self.num_machines*3+self.num_machines+8] == 0:
#             self.states[self.num_machines*3+self.num_machines-1] = 0  
#         elif self.states[self.num_machines*3+self.num_machines+8] % 720 == 0:
#             self.states[self.num_machines*3+self.num_machines-1] -= 15000

        # I: current solar irradiance
        self.states[self.num_machines*3+self.num_machines+6] = self.solar_irr[self.states[self.num_machines*3+self.num_machines+8]]
        # F: current wind speed
        self.states[self.num_machines*3+self.num_machines+7] = self.wind_sp[self.states[self.num_machines*3+self.num_machines+8]]

        
            
    def print_status(self):
        print("Buffer 0: ", self.buf0.amount)
        print("Buffer 1: ", self.buf1.amount)
        print("Buffer 2: ", self.buf2.amount)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import keras 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

file_si = "SolarIrradiance.csv"
file_ws = "WindSpeed.csv"
area = 300
e_solar = 0.2
cost_solar = 0.02
dt_solar = pd.read_csv(file_si)
solar_irr = np.array(dt_solar.iloc[:,2])*area*e_solar/1000

density_air = 1.225
radius = 20
power_coef = 0.593
e_gear = 0.9
e_elec = 0.9
cost_wind = 0.03
h = 3
dt_wind = pd.read_csv(file_ws)
wind_sp = (np.array(dt_wind.iloc[:,2])**3) * 0.5 * density_air * np.pi * (radius**2) * power_coef * e_gear * e_elec * h / 1000 
        

alpha = 1
lamb = 0.1


# print(actions[idx], len(actions[idx]))
# print(factory.states, len(factory.states))
# x = np.concatenate((actions[idx], factory.states))
# scaler = MinMaxScaler()
# x = scaler.fit_transform(np.reshape(x, [1,28]))
# print(x, x.shape)
# print(model.predict(x))
# model.train_on_batch(x_batch, y_batch)
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# classes = model.predict(x_test, batch_size=128)

yss = []
historys = []
weights = []
for r in range(1):

    #build the neural network, weights are stored in weights
    model = Sequential()
    model.add(Dense(32, input_dim=11+17, activation='sigmoid',kernel_initializer='normal'))
    model.add(Dense(32, activation='sigmoid', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam')

    factory = ManufacturingSystem(solar_irr=solar_irr, wind_sp=wind_sp)
    factory.mac0.set_debug(False)
    factory.mac1.set_debug(False)
    actions = factory.get_actions()

    ys = []
    history = []
    weight = []
    for n in range(100000):
        idx = np.random.randint(len(actions))
        alpha = 1/(n+1)
        
        #Set up the states and actions
        x = np.concatenate((actions[idx], factory.states))
        scaler  = MinMaxScaler()
        x = scaler.fit_transform(np.reshape(x, [1,28]))
        
        #current Q-value predicted by the neural network
        current =model.predict(x)
        prod_before = factory.states[factory.num_machines*3+factory.num_machines-1]
        
        #Output the current state, optimal policy and optimal cost 
        best_current= 999999
        best_action_current = None
        remainder=n % 1000
        if remainder==0:
            for a in actions:
                x_curr = np.concatenate((a, factory.states))
                x_curr = scaler.fit_transform(np.reshape(x_curr, [1,28]))
                predicted_reward_current = model.predict(x_curr)
                if predicted_reward_current < best_current:
                    best_current = predicted_reward_current
                    best_action_current = a
            print("Epoch: ", n)
            print("Current State: ", factory.states)
            print("Best Action: ", best_action_current, "Best Reward: ", best_current)
        
        #Calculate the reward for one step transition of MDP
        post, cost, actions = factory.on_action(actions[idx])
        prod = post[factory.num_machines*3+factory.num_machines-1] - prod_before
        reward = cost-prod
        
        #Find the optimal cost under current model for the Q-values in the next state
        bestFuture = 999999
        for a in actions:
            xp = np.concatenate((a, post))
            xp = scaler.fit_transform(np.reshape(xp, [1,28]))
            predicted_reward = model.predict(xp)
            if predicted_reward < bestFuture:
                bestFuture = predicted_reward
        
        #Update the new Q-value
        y = (1-alpha)*current+alpha*(reward+lamb*bestFuture)
        
        ys.append(y)

        #Calculate the difference in weights of the neural network
        #Calculate previous weights        
        pre_weight = []
        for layer in model.layers:
            w = layer.get_weights()
            for each in w:
                pre_weight += list(each.flatten())
        
        #Train the neural network
        history.append(model.train_on_batch(x, y))
        
        #Calculate the weights after training
        post_weight = []
        for layer in model.layers:
            w = layer.get_weights()
            for each in w:
                post_weight += list(each.flatten())
        
        #Difference in weights
        diff_weights = np.array(post_weight) - np.array(pre_weight)
        
        #L2 norm of the difference in weight vector are stored in weight
        weight.append(np.linalg.norm(diff_weights, 2))
        
        if remainder==0:
            print("Weight difference: ", weight[-1])
            print("----------------------------------------------")

        if n % 10000 == 0:
            plt.plot(weight[-10000:])
            plt.show()
    yss.append(np.reshape(ys, (len(ys),)))
    historys.append(history)
    weights.append(weight)


#Plot the weight difference
    
plt.figure(figsize=(12,8))
sns.tsplot(weights,time=range(len(weights[0])), ci=[68,95], condition="Weights")
plt.xlabel("Iteration")
plt.ylabel("Wight Difference")

plt.show()


