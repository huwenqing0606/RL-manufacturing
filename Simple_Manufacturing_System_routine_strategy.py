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
import mip

dt = 1
num_machines = 5
time_horizon = 100
capacity_of_buffer = [1000]*(num_machines-1)
#the buffermax
rated_power_of_machine = [99.0357/1000, 87.0517/1000, 91.7212/1000, 139.3991/1000, 102.8577/1000]
#rated power of machine measured in MegaWatt = 1000 kW
production_rate_of_machine = [1]*num_machines


def Mixed_Integer_Program(target_output):
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
    return prod_mat





"""
Testing for the Routine Strategy Selected by Mixed Integer Programming at Given Horizon
"""
def RoutineStrategy_Testing(number_iteration, #the number of testing iterations
                            target_output     #the target output
                            ):
    
    #open and output the results to the file routine_output.txt
    rtoutput = open('routine_output.txt', 'w')

    #Calculate and output the total cost, total throughput and total energy demand for mixed-integer programming with target output as the one given by the optimal strategy
    print("\n************************* Mixed Integer Programming with given Target Output *************************", file=rtoutput)
    print("***Run the system on routine policy by mixed-integer programming at a time horizon=", number_iteration,"***", file=rtoutput)
    target_output=int(target_output)
    print("Target Output =", target_output, file=rtoutput)
    routine_sol=Mixed_Integer_Program(target_output)
    print("Optimal solution from mixed-integer programming is given by \n", routine_sol.T, file=rtoutput)

    #close and save the results to the file
    rtoutput.close()

    return 0



"""
######################## MAIN TESTING FILE ##############################
######################## FOR DEBUGGING ONLY #############################

"""

if __name__=="__main__":
    #set the optimal production matrx which is a 0-1 matrix, rows=number_machines, columns=testing_number_iteration
    mipsol = open('sample_mixed_integer_programming_solution.txt', 'w')
    print("******************** Optimal Strategy for Simple Mixed-Integer Programming with Target Output from 0-100 ********************\n", file=mipsol)
    for i in [73]:
        print("\n------------- Target="+str(i)+" -------------")
        print("\n------------- Target="+str(i)+" -------------", file=mipsol)
        print("\nTarget=", i, file=mipsol)
        x=Mixed_Integer_Program(i)
        print("\noptimal solution is \n", x.T, file=mipsol)
    mipsol.close()        