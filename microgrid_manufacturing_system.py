"""
MDP for joint control of microgrid and manufactoring system
author: Wenqing Hu (Missouri S&T)
"""
import numpy as np
from random import choice

"""
Set up all parameters that are constant throughout the system
"""
Delta_t=1 
#the actual time measured in one decision epoch unit#
unit_reward_production=1
#the unit reward for each unit of production, i.e. the r^p, this applies to the end of the machine sequence
cutin_windspeed=0
#the cut-in windspeed (m/s), v^ci#
cutoff_windspeed=1000000
#the cut-off windspeed (m/s), v^co#
rated_windspeed=1
#the rated windspeed (m/s), v^r#
charging_discharging_efficiency=0.99
#the charging-discharging efficiency, eta#
rate_battery_discharge=1
#the rate for discharging the battery, b#
rate_consumption_charge=1
#the rate of consumption charge, r^c#
unit_operational_cost_solar=1
#the unit operational and maintanance cost for generating power from solar PV, r_omc^s#
unit_operational_cost_wind=1
#the unit operational and maintanance cost for generating power from wind turbine, r_omc^w#
unit_operational_cost_generator=1
#the unit opeartional and maintanance cost for generating power from generator, r_omc^g#
unit_operational_cost_battery=1
#the unit operational and maintanance cost for battery storage system per unit charging/discharging cycle, r_omc^b#
capacity_battery_storage=1
#the capacity of battery storage system, e#
SOC_max=10000000
#the maximum state of charge of battery system#
SOC_min=0
#the minimum state of charge of battery system#
area_solarPV=1
#the area of the solar PV system, a#
efficiency_solarPV=1
#the efficiency of the solar PV system, delta#
density_of_air=1
#calculate the rated power of the wind turbine, density of air, rho#
radius_wind_turbine_blade=1
#calculate the rated power of the wind turbine, radius of the wind turbine blade, r#
average_wind_speed=1
#calculate the rated power of the wind turbine, average wind speed, v_avg#
power_coefficient=1
#calculate the rated power of the wind turbine, power coefficient, theta#
gearbox_transmission_efficiency=1
#calculate the rated power of the wind turbine, gearbox transmission efficiency, eta_t#
electrical_generator_efficiency=1
#calculate the rated power of the wind turbine, electrical generator efficiency, eta_g#
rated_power_wind_turbine=0.5*density_of_air*np.pi*radius_wind_turbine_blade*radius_wind_turbine_blade*average_wind_speed*average_wind_speed*average_wind_speed*power_coefficient*gearbox_transmission_efficiency*electrical_generator_efficiency/1000
#the rated power of the wind turbine, RP_w#
number_windturbine=10
#the number of wind turbine in the onsite generation system, N_w#
number_generators=10
#the number of generators, n_g#
rated_output_power_generator=1
#the rated output power of the generator, G_p#
unit_reward_production=1
#unit reward for each unit of production, r^p#
unit_reward_soldbackenergy=1
#the unit reward from sold back energy, r^sb#
number_machines=5
#the total number of machines in the manufacturing system, total number of buffers=number_machines-1#


import pandas as pd
#read the solar irradiance and wind speed data from file#
file_SolarIrradiance = "SolarIrradiance.csv"
file_WindSpeed = "WindSpeed.csv"

data_solar = pd.read_csv(file_SolarIrradiance)
solarirradiance = np.array(data_solar.iloc[:,3])

data_wind = pd.read_csv(file_WindSpeed)
windspeed = np.array(data_wind.iloc[:,3])



"""
Define 3 major classes in the system: Machine, Buffer, Microgrid
"""
"""
the Machine class defines the variables and functions of one machine
"""
class Machine(object):
    def __init__(self,
                 name=1,
                 #the label of this machine#
                 lifetime_shape_parameter=1, 
                 #random lifetime of machine follows Weibull distribution with shape parameter lifetime_shape_parameter
                 lifetime_scale_parameter=1,
                 #random lifetime of machine follows Weibull distribution with scale parameter lifetime_scale_parameter
                 repairtime_mean=1,
                 #random repair time of machine follows exponential distribution with mean repairtime_mean
                 power_consumption_Opr=1,
                 #amount of power drawn by the machine if the machine state is Opr (Operating)
                 power_consumption_Idl=1,
                 #amount of power drawn by the machine if the machine state is Sta (Starvation) or Blo (Blockage), both are Idl (Idle) states
                 state="OFF",
                 #machine state can be "Opr" (Operating), "Blo" (Blockage), "Sta" (Starvation), "Off", "Brk" (Break)
                 control_action="K",
                 #control actions of machine, actions can be "K"-action (keep the original operational), "H"-action (to turn off the machine) or "W"-action (to turn on the machine)#
                 is_last_machine=False
                 #check whether or not the machine is the last machine in the queue, if it is last machine, then it contributes to the throughput#
                 ):
        self.name=name
        self.lifetime_shape_parameter=lifetime_shape_parameter
        self.lifetime_scale_parameter=lifetime_scale_parameter
        self.repairtime_mean=repairtime_mean
        self.power_consumption_Opr=power_consumption_Opr
        self.power_consumption_Idl=power_consumption_Idl
        self.unit_reward_production=unit_reward_production
        self.state=state
        self.control_action=control_action
        self.is_last_machine=is_last_machine
    
    def EnergyConsumption(self):
        #Calculate the energy consumption of one machine in a time unit#
        PC=0 
        #PC is the amount drawn by a machine in a time unit#
        if self.state=="Brk" or self.state=="Off":
            PC=0
        elif self.state=="Opr":
            PC=self.power_consumption_Opr*Delta_t
        elif self.state=="Sta" or self.state=="Blo":
            PC=self.power_consumption_Idl*Delta_t
        return PC

    def LastMachineProduction(self):
        #only the last machine will produce that contributes to the throughput, when the state is Opr and the control action is K#
        if self.is_last_machine:
            if self.state!="Opr" or self.control_action=="H":
                throughput=0
            elif self.state=="Opr" and self.control_action=="K":
                throughput=1
            else:
                throughput=0
        else:
            throughput=0
        return throughput*self.unit_reward_production
    
    def NextState_IsOff(self):
        #Based on the current state of the machine, determine if the state of the machine at next decision epoch is "Off"#
        #If is "Off" return True otherwise return False#
        #When return False, the next state lies in the set {"Brk", "Opr", "Sta", "Blo"}#
        if self.state=="Off":
            if self.control_action!="W":
                IsOff=True
            else:
                IsOff=False
        else:
            if self.control_action=="H":
                IsOff=True
            else:
                IsOff=False
        return IsOff
            
    def NextState_IsBrk(self):
        #Based on the current state of the machine, determine if the state of the machine at next decision epoch is "Brk"#
        #If is "Brk" return True otherwise return False#
        #When return False, the next state lies in the set {"Opr", "Sta", "Blo", "Off"}#
        L=np.random.weibull(self.lifetime_shape_parameter, self.lifetime_scale_parameter)
        #the random variable L is the lifetime#
        D=np.random.exponential(1/self.repairtime_mean)
        #the random variable D is the repair time# 
        if self.state=="Brk":
            if D>=Delta_t:
                IsBrk=True
            else:
                IsBrk=False
        else:
            if self.state!="Off":
                if L<Delta_t:
                    IsBrk=True
                else:
                    IsBrk=False
            else:
                IsBrk=False
        return IsBrk
    
    def PrintMachine(self):
        #print the status of the current machine: state, control_action taken, Energy Consumption, throughput, decide whether the next machine state is Brk#
        print("Machine", self.name, "=", self.state, ",", "action=", self.control_action)
        print(" Energy Consumption=", self.EnergyConsumption())
        if self.is_last_machine:
            print(" ")
            print(" throughput=", self.LastMachineProduction())
        return " "
        
        
        
"""
the Buffer class defines variables and functions of one buffer
"""
class Buffer(object):
    def __init__(self, 
                 name=1,
                 #the label of this buffer#
                 state=0,
                 #the buffer state is an integer from buffer_min (=0) to buffer_max 
                 buffer_max=1, 
                 #the maximal capacity of the buffer#
                 buffer_min=0,
                 #the minimal capacity of the buffer is zero#
                 previous_machine_state="Opr",
                 #the state of the machine that is previous to the current buffer#
                 next_machine_state="Off",
                 #the state of the machine that is next to the current buffer#
                 previous_machine_control_action="K",
                 #the control action applied to the machine that is previous to the current buffer#
                 next_machine_control_action="K"
                 #the control action applied to the machine that is next to the current buffer#
                 ):
        self.name=name
        self.state=state
        self.buffer_max=buffer_max
        self.buffer_min=buffer_min
        self.previous_machine_state=previous_machine_state
        self.next_machine_state=next_machine_state
        self.previous_machine_control_action=previous_machine_control_action
        self.next_machine_control_action=next_machine_control_action
        
    def NextState(self):
        #calculate the state of the buffer at next decision epoch, return this state#
        nextstate=self.state
        if self.previous_machine_state!="Opr" or self.previous_machine_control_action=="H":
            I_previous=0
        elif self.previous_machine_state=="Opr" and self.previous_machine_control_action=="K":
            I_previous=1
        else:
            I_previous=0
        if self.next_machine_state!="Opr" or self.next_machine_control_action=="H":
            I_next=0
        elif self.next_machine_state=="Opr" and self.next_machine_control_action=="K":
            I_next=1
        else:
            I_next=0
        nextstate=nextstate+I_previous-I_next
        if nextstate>self.buffer_max:
            nextstate=self.buffer_max
        if nextstate<self.buffer_min:
            nextstate=self.buffer_min
        return nextstate

    def PrintBuffer(self):
        #print the status of the current buffer: buffer state, next buffer state#
        print("Buffer", self.name, "=", self.state)
        return " "


        
"""
the Microgrid class defines variables and functions of the microgrid
"""
class Microgrid(object):
    def __init__(self,
                 workingstatus=[0,0,0],
                 #the working status of [solar PV, wind turbine, generator]#
                 SOC=0,
                 #the state of charge of the battery system#
                 actions_adjustingstatus=[0,0,0],
                 #the actions of adjusting the working status (connected =1 or not =0 to the load) of the [solar, wind, generator]#
                 actions_solar=[0,0,0],
                 #the solar energy used for supporting [manufaturing, charging battery, sold back]#
                 actions_wind=[0,0,0],
                 #the wind energy used for supporting [manufacturing, charging battery, sold back]#
                 actions_generator=[0,0,0],
                 #the use of the energy generated by the generator for supporting [manufacturing, charging battery, sold back]#
                 actions_purchased=[0,0],
                 #the use of the energy purchased from the grid for supporting [manufacturing, charging battery]#
                 actions_discharged=0,
                 #the energy discharged by the battery for supporting manufacturing#
                 solarirradiance=0,
                 #the environment feature: solar irradiance at current decision epoch#
                 windspeed=0
                 #the environment feature: wind speed at current decision epoch#
                 ):
        self.workingstatus=workingstatus
        self.SOC=SOC
        self.actions_adjustingstatus=actions_adjustingstatus
        self.actions_solar=actions_solar
        self.actions_wind=actions_wind
        self.actions_generator=actions_generator
        self.actions_purchased=actions_purchased
        self.actions_discharged=actions_discharged
        self.solarirradiance=solarirradiance
        self.windspeed=windspeed
        
    def transition(self):
        workingstatus=self.workingstatus
        SOC=self.SOC
        if self.actions_adjustingstatus[1-1]==1:
            workingstatus[1-1]=1
        else:
            workingstatus[1-1]=0
        #determining the next decision epoch working status of solar PV, 1=working, 0=not working#
        if self.actions_adjustingstatus[2-1]==0 or self.windspeed>cutoff_windspeed or self.windspeed<cutin_windspeed:
            workingstatus[2-1]=0
        else: 
            if self.actions_adjustingstatus[2-1]==1 and self.windspeed<=cutoff_windspeed and self.windspeed>=cutin_windspeed:
                workingstatus[2-1]=1
        #determining the next decision epoch working status of wind turbine, 1=working, 0=not working#        
        if self.actions_adjustingstatus[3-1]==1:
            workingstatus[3-1]=1
        else:
            workingstatus[3-1]=0
        #determining the next decision epoch working status of generator, 1=working, 0=not working#
        SOC=self.SOC+(self.actions_solar[2-1]+self.actions_wind[2-1]+self.actions_generator[2-1]+self.actions_purchased[2-1])*charging_discharging_efficiency-self.actions_discharged/charging_discharging_efficiency
        if SOC>SOC_max:
            SOC=SOC_max
        if SOC<SOC_min:
            SOC=SOC_min
        #determining the next desicion epoch SOC, state of charge of the battery system#
        return workingstatus, SOC
    
    def EnergyConsumption(self):
        #returns the energy consumption from the grid#
        return -(self.actions_solar[1-1]+self.actions_wind[1-1]+self.actions_generator[1-1]+self.actions_discharged)

    def OperationalCost(self):
        #returns the operational cost for the onsite generation system#
        if self.workingstatus[1-1]==1:
            energy_generated_solar=self.solarirradiance*area_solarPV*efficiency_solarPV/1000
        else:
            energy_generated_solar=0
        #calculate the energy generated by the solar PV, e_t^s#
        if self.workingstatus[2-1]==1 and self.windspeed<rated_windspeed and self.windspeed>=cutin_windspeed:
            energy_generated_wind=number_windturbine*rated_power_wind_turbine*(self.windspeed-cutin_windspeed)/(rated_windspeed-cutin_windspeed)
        else:
            if self.workingstatus[2-1]==1 and self.windspeed<cutoff_windspeed and self.windspeed>=rated_windspeed:
                energy_generated_wind=number_windturbine*rated_power_wind_turbine*Delta_t
            else:
                energy_generated_wind=0
        #calculate the energy generated by the wind turbine, e_t^w#
        if self.workingstatus[3-1]==1:
            energy_generated_generator=number_generators*rated_output_power_generator*Delta_t
        else:
            energy_generated_generator=0
        #calculate the energy generated bv the generator, e_t^g#
        operational_cost=energy_generated_solar*unit_operational_cost_solar+energy_generated_wind*unit_operational_cost_wind+energy_generated_generator*unit_operational_cost_generator
        operational_cost+=(self.actions_discharged+self.actions_solar[2-1]+self.actions_wind[2-1]+self.actions_generator[2-1])*Delta_t*unit_operational_cost_battery/(2*capacity_battery_storage*(SOC_max-SOC_min))
        #calculate the operational cost for the onsite generation system#
        return operational_cost
    
    def SoldBackReward(self):
        #calculate the sold back reward (benefit)#
        return (self.actions_solar[3-1]+self.actions_wind[3-1]+self.actions_generator[3-1])*unit_reward_soldbackenergy
    
    def PrintMicrogrid(self):
        #print the current and the next states of the microgrid#
        print("Microgrid working status [solar PV, wind turbine, generator]=", self.workingstatus, ", SOC=", self.SOC)
        print(" microgrid actions [solar PV, wind turbine, generator]=", self.actions_adjustingstatus)
        print(" solar energy supporting [manufaturing, charging battery, sold back]=", self.actions_solar)
        print(" wind energy supporting [manufacturing, charging battery, sold back]=", self.actions_wind)
        print(" generator energy supporting [manufacturing, charging battery, sold back]=", self.actions_generator)
        print(" energy purchased from grid supporting [manufacturing, charging battery]=", self.actions_purchased)
        print(" energy discharged by the battery supporting manufacturing=", self.actions_discharged)
        print(" solar irradiance=", self.solarirradiance)
        print(" wind speed=", self.windspeed)
        print(" Microgrid Energy Consumption=", self.EnergyConsumption())
        print(" Microgrid Operational Cost=", self.OperationalCost())
        print(" Microgrid SoldBackReward=", self.SoldBackReward())
        return " "


"""    
Combining the above three classes, define the variables and functions for the whole manufacturing system
"""
class ManufacturingSystem(object):
    def __init__(self,
                 machine_states,
                 #set the machine states for all machines in the manufacturing system#
                 machine_control_actions,
                 #set the control actions for all machines in the manufacturing system#
                 buffer_states,
                 #set the buffer states for all buffers in the manufacturing system#
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
                 #set the microgrid states and control actions#
                 ):
        self.machine_states=machine_states
        self.machine_control_actions=machine_control_actions
        self.buffer_states=buffer_states
        #initialize all machines, ManufacturingSystem.machine=[Machine1, Machine2, ..., Machine_{number_machines}]#
        self.machine=[]
        for i in range(number_machines):
            if i!=number_machines-1:
                self.machine.append(Machine(name=i+1, 
                                            state=self.machine_states[i], 
                                            control_action=self.machine_control_actions[i], 
                                            is_last_machine=False))
            else:
                self.machine.append(Machine(name=i+1, 
                                            state=self.machine_states[i], 
                                            control_action=self.machine_control_actions[i], 
                                            is_last_machine=True))
        #initialize all buffers, ManufacturingSystem.buffer=[Buffer1, Buffer2, ..., Buffer_{numbers_machines-1}]
        self.buffer=[]
        for j in range(number_machines-1):
            self.buffer.append(Buffer(name=j+1, 
                                      state=self.buffer_states[j], 
                                      previous_machine_state=self.machine[j].state, 
                                      next_machine_state=self.machine[j+1].state,
                                      previous_machine_control_action=self.machine[j].control_action,
                                      next_machine_control_action=self.machine[j+1].control_action
                                      ))
        self.grid=grid
        
    def transition_manufacturing(self):
        #based on current states and current control actions of the whole manufacturing system, calculate states at the the next decision epoch#
        #states include machine states, buffer states and microgrid states#
        buffer_states=[]
        for j in range(number_machines-1):
            buffer_states.append(self.buffer[j].NextState())
        #based on current machine states and control actions taken, calculate the next states of all buffers#    
        Off=[]
        Brk=[]
        Sta=[]
        Blo=[]
        #Set up four 0/1 sequence that test the next states being "Off", "Brk", "Sta" or "Blo". If none of these, then "Opr"#
        for i in range(number_machines):
            Off.append(0)
            Brk.append(0)
            Sta.append(0)
            Blo.append(0)
        for i in range(number_machines):
        #Check the possibilities of "Off" or "Brk" states#    
            if self.machine[i].NextState_IsOff():
                Off[i]=1
            if self.machine[i].NextState_IsBrk():
                Brk[i]=1
        for i in range(number_machines):
        #Check the possibilities of "Sta" states#
            if i==0:
                Sta[i]=0
            else:
                if Brk[i]==1 or Off[i]==1:
                    Sta[i]=0
                else:
                    if buffer_states[i-1]==self.buffer[i-1].buffer_min:
                        if Brk[i-1]==1 or Sta[i-1]==1 or Off[i-1]==1:
                            Sta[i]=1
                        else:
                            Sta[i]=0
                    else:
                        Sta[i]=0
        for i in reversed(range(number_machines)):
        #Check the possibilities of "Blo" states#
            if i==number_machines-1:
                Blo[i]=0
            else:
                if Brk[i]==1 or Off[i]==1:
                    Blo[i]=0
                else:
                    if buffer_states[i]==self.buffer[i].buffer_max:
                        if Brk[i+1]==1 or Blo[i+1]==1 or Off[i+1]==1:
                            Blo[i]=1
                        else:
                            Blo[i]=0
                    else:
                        Blo[i]=0
        #based on current machine states and control actions taken, calculate the next states of all machines#    
        machine_states=[]                
        for i in range(number_machines):
            if Off[i]==1:
                machine_states.append("Off")
            elif Brk[i]==1:
                machine_states.append("Brk")
            elif Sta[i]==1:
                machine_states.append("Sta")
            elif Blo[i]==1:
                machine_states.append("Blo")
            else: 
                machine_states.append("Opr")
        #return the new states#
        return machine_states, buffer_states

    def average_total_cost(self):
        #calculate the average total cost of the manufacturing system, E(S,A), based on the current machine, buffer, microgrid states and actions#
        E_mfg=0
        #total energy consumed by the manufacturing system, summing over all machines#
        for i in range(number_machines):
            E_mfg=E_mfg+self.machine[i].EnergyConsumption()
        #the energy consumption cost#            
        TF=(E_mfg+self.grid.EnergyConsumption())*rate_consumption_charge
        #the operational cost for the microgrid system#
        MC=self.grid.OperationalCost()
        #the prduction throughput of the manufacturing system#
        TP=self.machine[number_machines-1].LastMachineProduction()*unit_reward_production
        #the sold back reward#
        SB=self.grid.SoldBackReward()
        return TF+MC-TP-SB


"""
Simulate admissible actions based on the current state S_{t+1} of the manufacturing system, 
the admissible actions are A_{t+1}=(A^d, A^c, A^r)
"""
class ActionSimulation(object):
    def __init__(self,
                 System=ManufacturingSystem(machine_states=["Off", "Off", "Off", "Off", "Off"],
                                            machine_control_actions=["K", "K", "K", "K", "K"],
                                            buffer_states=[0,0,0,0],
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
                                                           )),
                 theta=[0,0,0,0,0,0]):
        #the ManufacturingSystem is with new states S_{t+1} but old actions A_{t}, we obtain the admissible A_{t+1} in this class#
        self.System=System
        #theta is the proportionality parameters theta=[lambda_s^m, lambda_s^b, lambda_w^m, lambda_w^b, lambda_g^m, lambda_g^]#
        self.theta=theta         
    
    def MachineActions(self):
        #Based on current machine states in the system, randomly uniformly simulate an admissible action for all machines#
        machine_actions=[]
        for i in range(number_machines):
            if self.System.machine_states[i]=="Opr":
                machine_actions.append(choice(["K", "H"]))
            elif self.System.machine_states[i]=="Blo":
                machine_actions.append(choice(["K", "H"]))
            elif self.System.machine_states[i]=="Sta":
                machine_actions.append(choice(["K", "H"]))
            elif self.System.machine_states[i]=="Off":
                machine_actions.append(choice(["K", "W"]))
            else:
                machine_actions.append("K")
        return machine_actions
    
    def MicroGridActions_adjustingstatus(self):
        #randomly uniformly simulate an action that adjusts the status (connected=1) of the microgrid [solar, wind, generator]#
        actions_adjustingstatus=[]
        for i in range(3):
            actions_adjustingstatus.append(choice([0,1]))
        return actions_adjustingstatus
    
    def MicroGridActions_SolarWindGenerator(self, theta):
        #from the updated proportionality parameter theta return the corresponding actions on solar, wind and generator#
        if self.System.grid.workingstatus[1-1]==1:
            energy_generated_solar=self.System.grid.solarirradiance*area_solarPV*efficiency_solarPV/1000
        else:
            energy_generated_solar=0
        #calculate the energy generated by the solar PV, e_t^s#
        if self.System.grid.workingstatus[2-1]==1 and self.System.grid.windspeed<rated_windspeed and self.System.grid.windspeed>=cutin_windspeed:
            energy_generated_wind=number_windturbine*rated_power_wind_turbine*(self.System.grid.windspeed-cutin_windspeed)/(rated_windspeed-cutin_windspeed)
        else:
            if self.System.grid.workingstatus[2-1]==1 and self.System.grid.windspeed<cutoff_windspeed and self.System.grid.windspeed>=rated_windspeed:
                energy_generated_wind=number_windturbine*rated_power_wind_turbine*Delta_t
            else:
                energy_generated_wind=0
        #calculate the energy generated by the wind turbine, e_t^w#
        if self.System.grid.workingstatus[3-1]==1:
            energy_generated_generator=number_generators*rated_output_power_generator*Delta_t
        else:
            energy_generated_generator=0
        #given the new theta, calculated the actions_solar, actions_wind, actions_generator#
        actions_solar=[energy_generated_solar*theta[1-1], energy_generated_solar*theta[2-1], energy_generated_solar*(1-theta[1-1]-theta[2-1])]
        actions_wind=[energy_generated_wind*theta[3-1], energy_generated_wind*theta[4-1], energy_generated_wind*(1-theta[3-1]-theta[4-1])]
        actions_generator=[energy_generated_generator*theta[5-1], energy_generated_generator*theta[6-1], energy_generated_generator*(1-theta[5-1]-theta[6-1])]
        return actions_solar, actions_wind, actions_generator
    
    def MicroGridActions_PurchasedDischarged(self, 
                                             actions_solar=[0,0,0],
                                             actions_wind=[0,0,0],
                                             actions_generator=[0,0,0]):
        #randomly simulate an action that determines the use of the purchased energy and the energy discharge#
        #actions_solar, actions_wind, actions_generator are the actions to be taken at current system states#
        TotalSoldBack=actions_solar[3-1]+actions_wind[3-1]+actions_generator[3-1]
        #Total amount of sold back energy#
        TotalBattery=actions_solar[2-1]+actions_wind[2-1]+actions_generator[2-1]
        #Total amount if energy charged to the battery#
        SOC_Condition=self.System.grid.SOC-rate_battery_discharge*Delta_t/charging_discharging_efficiency-SOC_min
        #The condition for SOC at the current system state#
        E_mfg=0
        for i in range(number_machines):
            E_mfg=E_mfg+self.System.machine[i].EnergyConsumption()
        #total energy consumed by the manufacturing system, summing over all machines#
        p_hat=E_mfg-(actions_solar[1-1]+actions_wind[1-1]+actions_generator[1-1])
        if p_hat<0:
            p_hat=0
        #Set the p_hat#
        p_tilde=E_mfg-(actions_solar[1-1]+actions_wind[1-1]+actions_generator[1-1]+rate_battery_discharge*Delta_t)
        if p_tilde<0:
            p_tilde=0
        #Set the p_tilde#
        ####Calculate actions_purchased and actions_discharged according to the table in the paper####
        actions_purchased=[0,0]
        actions_discharged=0
        if TotalSoldBack>0 and TotalBattery>0 and SOC_Condition>0:
            actions_purchased=[0,0]
            actions_discharged=0
        elif TotalSoldBack>0 and TotalBattery>0 and SOC_Condition<=0:
            actions_purchased=[0,0]
            actions_discharged=0
        elif TotalSoldBack>0 and TotalBattery<=0 and SOC_Condition>0:
            actions_purchased=[0,0]
            actions_discharged=choice([0, rate_battery_discharge*Delta_t])
        elif TotalSoldBack>0 and TotalBattery<=0 and SOC_Condition<=0:
            actions_purchased=[0,0]
            actions_discharged=0
        elif TotalSoldBack<=0 and TotalBattery>0 and SOC_Condition>0:
            actions_purchased[2-1]=choice([0, p_hat])
            actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            actions_discharged=0
        elif TotalSoldBack<=0 and TotalBattery>0 and SOC_Condition<=0:
            actions_purchased[2-1]=choice([0, p_hat])
            actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            actions_discharged=0
        elif TotalSoldBack<=0 and TotalBattery<=0 and SOC_Condition>0:
            actions_discharged=choice([0, rate_battery_discharge*Delta_t])
            if actions_discharged==0:
                actions_purchased[2-1]=choice([0, p_hat])
                actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            else:
                actions_purchased[2-1]=0
                actions_purchased[1-1]=p_tilde
        else:
            actions_purchased[2-1]=choice([0, p_hat])
            actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            actions_discharged=0
            
        return actions_purchased, actions_discharged
            


"""
testing on random admissible actions
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
    System=ManufacturingSystem(machine_states=["Opr", "Opr", "Off", "Opr", "Opr"],
                               machine_control_actions=["K", "K", "K", "K", "K"],
                               buffer_states=[0,0,0,0],
                               grid=grid
                               )
    theta=[0,0,0,0,0,0]
    for t in range(1000):
        #current states and actions S_t and A_t are stored in class System#
        print("*********************Time Step", t, "*********************")
        for i in range(number_machines):
            print(System.machine[i].PrintMachine())
            if i!=number_machines-1:
                print(System.buffer[i].PrintBuffer())
        print(System.grid.PrintMicrogrid())
        print("Average Total Cost=", System.average_total_cost())
        #update the theta#
        theta_sum=np.random.uniform(0,1,size=None)
        theta_proportion=np.random.uniform(0,1,size=None)
        theta=[theta_sum*theta_proportion, theta_sum*(1-theta_proportion), theta_sum*theta_proportion, theta_sum*(1-theta_proportion), theta_sum*theta_proportion, theta_sum*(1-theta_proportion)]
        #calculate the next states and actions, S_{t+1}, A_{t+1}#        
        next_machine_states, next_buffer_states=System.transition_manufacturing()
        next_workingstatus, next_SOC=System.grid.transition()
        next_action=ActionSimulation(System=ManufacturingSystem(machine_states=next_machine_states,
                                                                machine_control_actions=["K", "K", "K", "K", "K"],
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

