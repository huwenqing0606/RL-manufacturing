# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:38:06 2020

@author: huwenqing

Title: Projection operator onto the 2-dim simplex {(\lambda_1, \lambda_2): \lambda_1\geq 0, \lambda_2\geq 0, 0\leq \lambda_1+\lambda_2\leq 1}
"""
import numpy as np
import operator
from functools import reduce
import matplotlib.pyplot as plt

def projectionSimplex(theta_0):
    theta_proj=theta_0
    if theta_0[1-1]<0 and theta_0[2-1]<0:
        theta_proj=[0, 0]
    elif theta_0[1-1]>=0 and theta_0[1-1]<=1 and theta_0[2-1]<0:
        theta_proj=[theta_0[1-1], 0]
    elif theta_0[1-1]>1 and theta_0[1-1]-theta_0[2-1]>1:
        theta_proj=[1, 0]
    elif theta_0[1-1]>0 and theta_0[1-1]+theta_0[2-1]>=1 and theta_0[2-1]>0 and theta_0[1-1]-theta_0[2-1]>=-1 and theta_0[1-1]-theta_0[2-1]<=1: 
        theta_proj=[(1+theta_0[1-1]-theta_0[2-1])/2, (1-theta_0[1-1]+theta_0[2-1])/2]
    elif theta_0[2-1]>1 and theta_0[1-1]-theta_0[2-1]<-1:
        theta_proj=[0, 1]
    elif theta_0[2-1]>=0 and theta_0[2-1]<=1 and theta_0[1-1]<0:
        theta_proj=[0, theta_0[2-1]]
    else:
        theta_proj=theta_0
    return theta_proj


def projection(theta):
    projection=[projectionSimplex([theta[1-1], theta[2-1]]), projectionSimplex([theta[3-1], theta[4-1]]), projectionSimplex([theta[5-1], theta[6-1]])]
    return reduce(operator.add, projection)


if __name__=="__main__":
 
    x = [[0, 0], [0, 1], [1, 0]] 
    y = [[0, 1], [1, 0], [0, 0]]

    for i in range(len(x)): 
        plt.plot(x[i], y[i], color='g')

    theta=np.random.uniform(-1,1,size=6)
    print(theta)
    print(projection(theta))
    x = [[theta[0], projection(theta)[0]], [theta[2], projection(theta)[2]], [theta[4], projection(theta)[4]]] 
    y = [[theta[1], projection(theta)[1]], [theta[3], projection(theta)[3]], [theta[5], projection(theta)[5]]]

    for i in range(len(x)): 
        plt.plot(x[i], y[i], color='r')
        plt.scatter(x[i], y[i], color='b')

    plt.xlim(-1,2)
    plt.ylim(-1,2)
    plt.show()
