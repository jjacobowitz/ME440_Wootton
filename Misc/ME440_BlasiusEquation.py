# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:12:13 2021

@author: Jared
"""
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

def Blasius(eta, _f):
    f = _f[0]
    fp = _f[1]
    fpp = _f[2]
    fppp = -f*fpp
    return fp,fpp,fppp

def optimize(fpp0):
    sol = solve_ivp(Blasius, (0,10), (0,0,fpp0), t_eval=np.linspace(0,10,100))
    return sol.y[1][-1] - 1

fpp0 = root_scalar(optimize, x0=0, x1=1)
print(fpp0)

sol = solve_ivp(Blasius, (0,10), (0,0,fpp0.root), t_eval=np.linspace(0,10,100))

plt.figure()
plt.plot(sol.y[1], sol.t)
plt.ylabel("$eta$")
plt.xlabel("$f'$")
plt.title("Blasius Equation")
plt.show()