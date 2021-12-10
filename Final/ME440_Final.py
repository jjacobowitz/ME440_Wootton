# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
ME440 Advanced Fluid Mechanics
Final Exam

Problem 3 code
"""
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

plt.close("all")

kappa = 0.41
B = 5.0
rho = 1.2           # kg/m^3; density of air
mu = 1.8e-5         # kg/m/s; dynamics viscosity of air
nu = mu/rho         # m^2/s; kinematic viscosity of air
h = 0.01            # m; half gap width
Re = 1e4            # Reynolds number


# def Cf_implicit(Cf, Re, h, nu, kappa, B):
#     return np.sqrt(8/Cf) - 2*np.log(Re*np.sqrt(Cf/8))/kappa - B


# Cf = root_scalar(Cf_implicit, args=(Re, h, nu, kappa, B), x0=0.1, x1=1)
# print(Cf)


def Cf_implicit(Cf, Re, h, nu, kappa, B):
    return 8/(2*np.log(Re*np.sqrt(Cf/8))/kappa + 2*B)**2
    # return Re/4/(np.log(np.sqrt(Re*Cf)/2)/kappa + B)**2


Cf_old = 100
Cf = Cf_implicit(Cf_old, Re, h, nu, kappa, B)
while abs((Cf_old-Cf)/Cf_old) > 1e-6:
    Cf_old, Cf = Cf, Cf_implicit(Cf, Re, h, nu, kappa, B)
print(Cf)

U = Re*nu/h
tau_w = Cf*rho*U**2/8
vstar = np.sqrt(tau_w/rho)


def u(y, vstar, nu, h, kappa, B):
    if -h < y < 0:
        r = y + h
        return vstar*(np.log(vstar*r/nu)/kappa + B)
    else:
        r = h - y
        return U - (vstar*(np.log(vstar*r/nu)/kappa + B))


y = np.linspace(-h+1e-10, h-1e-10, 1000)
plt.figure()
uy = [u(_y, vstar, nu, h, kappa, B) for _y in y]
plt.plot(uy, y)
# plt.plot(tau_w*y, y)
plt.ylim(-h, h)
plt.xlabel(r"Velocity ($\bar{u}(y)$)")
plt.ylabel(r"Position ($y$)")
plt.tight_layout()
plt.show()


def mu_T(y, vstar, nu):
    if -h < y < 0:
        return vstar/nu/y
    else:
        return -vstar/nu/y


plt.figure()
mu_Ty = [mu_T(_y, vstar, nu) for _y in y]
plt.plot(mu_Ty, y)
plt.show()
