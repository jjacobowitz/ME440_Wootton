# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
ME440 Advanced Fluid Mechanics

Problem 3-20 of the `Viscous Fluid` textbook for Homework 4
"""
import numpy as np
from scipy.optimize import root_scalar

def func(t):
    nu = 1.5e-5             # m^2/s; kinematic viscosity
    h = 0.02                # m; gap width
    U = 0.3                 # m/s; plate velocity
    u = 0.14                # m/s; centerline velocity
    y = h/2                 # m; centerline height
    
    t_star = nu*t/h**2      # dimensionless time
    
    summation = 0
    for n in range(1,100):
        summation += 1/n*np.exp(-(n**2)*np.pi**2*t_star)*np.sin(n*np.pi*y/h)
    
    return u/U - ((1-y/h) - 2/np.pi*summation)

print(root_scalar(func, x0=1, x1=2))




