"""
MDP for joint control of microgrid and manufactoring system
author: Wenqing Hu (Missouri S&T)
"""
import numpy as np


"""
Set up all parameters that are constant throughout the system
"""
T=60
#the total number of decision epoches in a whole period#
Delta_t=1 
#the actual time measured in one decision epoch unit#
unit_reward_production=1
#the unit reward for each unit of production, i.e. the r^p, this applies to the end of the machine sequence
cutin_windspeed=1
#the cut-in windspeed (m/s), v^ci#
cutoff_windspeed=2
#the cut-off windspeed (m/s), v^co#
rated_windspeed=1.5
#the rated windspeed (m/s), v^r#
charging_discharging_efficiency=1
#the charging-discharging efficiency, eta#
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
SOC_max=1000
#the maximum state of charge of battery system#
SOC_min=1
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
unit_reward_soldbackenergy=1
#the unit reward from sold back energy, r^sb#
number_machines=3
#the total number of machines in the manufacturing system, total number of buffers=number_machines-1#


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
        throughput=0
        if self.is_last_machine:
            if self.state!="Opr" or self.control_action=="H":
                throughput=0
            elif self.state=="Opr" and self.control_action=="K":
                throughput=1
        return throughput*self.unit_reward_production
    
    def NextState_IsBrk(self):
        #Based on the current state of the machine, determine if the state of the machine at next decision epoch is "Brk"#
        #If is "Brk" return True otherwise return False#
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
            if L<Delta_t:
                IsBrk=True
            else:
                IsBrk=False
        return IsBrk
    
    def PrintMachine(self):
        #print the status of the current machine: state, control_action taken, Energy Consumption, throughput, decide whether the next machine state is Brk#
        print("Machine ", self.name, ": ")
        print("state = ", self.state)
        print("control action taken = ", self.control_action)
        print("Energy Consumption = ", self.EnergyConsumption())
        print("is last machine (T/F) = ", self.is_last_machine)
        print("throughtput = ", self.LastMachineProduction())
        print("next state is Brk (T/F) = ", self.NextState_IsBrk())    
        return "***Machine printed***"
        
        
        
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
        if self.next_machine_state!="Opr" or self.next_machine_control_action=="H":
            I_next=0
        elif self.next_machine_state=="Opr" and self.next_machine_control_action=="K":
            I_next=1
        nextstate=nextstate+I_previous-I_next
        return nextstate

    def PrintBuffer(self):
        #print the status of the current buffer: buffer state, next buffer state#
        print("Buffer ", self.name, ": ")
        print("buffer state = ", self.state)
        print("next buffer state = ", self.NextState())
        return "***Buffer printed***"
        
"""
the Microgrid class defines variables and functions of the microgrid
"""
class Microgrid(object):
    def __init__(self,
                 workingstatus=[0,0,0],
                 #the working status of [solar PV, wind turbine, generator]#
                 SOC=0,
                 #the state of charge of the battery system#
                 actions_adjustingstatus=[0,1,0],
                 #the actions of adjusting the working status (connected =1 or not =0 to the load) of the [solar, wind, generator]#
                 actions_solar=[0,1,0],
                 #the solar energy used for supporting [manufaturing, charging battery, sold back]#
                 actions_wind=[0,0,1],
                 #the wind energy used for supporting [manufacturing, charging battery, sold back]#
                 actions_generator=[1,0,0],
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
        if self.actions_adjustingstatus[1-1]==1:
            self.workingstatus[1-1]=1
        else:
            self.workingstatus[1-1]=0
        #determining the next decision epoch working status of solar PV, 1=working, 0=not working#
        if self.actions_adjustingstatus[2-1]==0 or self.windspeed>cutin_windspeed or self.windspeed<cutoff_windspeed:
            self.workingstatus[2-1]=0
        else: 
            if self.actions_adjustingstatus[2-1]==1 and self.windspeed<=cutin_windspeed and self.windspeed>=cutoff_windspeed:
                self.workingstatus[2-1]=1
        #determining the next decision epoch working status of wind turbine, 1=working, 0=not working#        
        if self.actions_adjustingstatus[3-1]==1:
            self.workingstatus[3-1]=1
        else:
            self.workingstatus[3-1]=0
        #determining the next decision epoch working status of generator, 1=working, 0=not working#
        self.SOC=self.SOC+(self.actions_solar[2-1]+self.actions_wind[2-1]+self.actions_generator[2-1]+self.actions_purchased[2-1])*charging_discharging_efficiency-self.actions_discharged/charging_discharging_efficiency
        #update the SOC, state of charge of the battery system#
    
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
        energy_generated_generator=1
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
        print("Microgrid working status = ", self.workingstatus)
        print("SOC = ", self.SOC)
        print("the actions of adjusting the working status = ", self.actions_adjustingstatus)
        print("the solar energy used for supporting [manufaturing, charging battery, sold back] = ", self.actions_solar)
        print("the wind energy used for supporting [manufacturing, charging battery, sold back] = ", self.actions_wind)
        print("the use of the energy generated by the generator for supporting [manufacturing, charging battery, sold back] = ", self.actions_generator)
        print("the use of the energy purchased from the grid for supporting [manufacturing, charging battery] = ", self.actions_purchased)
        print("the energy discharged by the battery for supporting manufacturing = ", self.actions_discharged)
        print("the environment feature: solar irradiance at current decision epoch = ", self.solarirradiance)
        print("the environment feature: wind speed at current decision epoch = ", self.windspeed)
        print("Microgrid Energy Consunption = ", self.EnergyConsumption())
        print("Microgrid Operational Cost = ", self.OperationalCost())
        print("Microgrid SoldBackReward = ", self.SoldBackReward())
        return "***Microgrid printed***"


"""    
Combining the above three classes, define the variables and functions for the whole manufacturing system
"""
class ManufacturingSystem(object):
    def __init__(self,
                 machine_states=["Opr", "Off", "Brk"],
                 #set the machine states for all machines in the manufacturing system#
                 machine_control_actions=["K", "K", "W"],
                 #set the control actions for all machines in the manufacturing system#
                 buffer_states=[0,0]
                 #set the buffer states for all buffers in the manufacturing system#
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
    
    def transition(self):
        #based on current states and current control actions of the whole manufacturing system, calculate states at the the next decision epoch#
        #states include machine states, buffer states and microgrid states#
        for j in range(number_machines-1):
        #based on current machine states and control actions taken, calculate the next states of all buffers#    
            self.buffer[j].state=self.buffer[j].NextState()
        for i in range(number_machines):
            if self.machine[i].NextState_IsBrk():
                self.machine[i].state="Brk"
            else:
                if i==0:
                    self.machine[i].state=
                



    
if __name__ == "__main__":
    ManufacturingSystem=ManufacturingSystem()
    for i in range(number_machines):
        print("----------------- i=", i, "-----------------")
        print(ManufacturingSystem.machine[i].PrintMachine())
        if i!=number_machines-1:
            print(ManufacturingSystem.buffer[i].PrintBuffer())
    ManufacturingSystem.transition()
    print("----------------- One Step Transition -----------------")
    for i in range(number_machines):
        print("----------------- i=", i, "-----------------")
        print(ManufacturingSystem.machine[i].PrintMachine())
        if i!=number_machines-1:
            print(ManufacturingSystem.buffer[i].PrintBuffer())
        
