# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
21 November 2021 (Fall 2021)
ME440 Advanced Fluid Mechanics

Solving for the conditions of the Falkner-Skan equation for boundary layer flow
"""
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")


def get_beta():
    while True:
        try:
            beta = float(input("Enter a value for beta: "))
        except ValueError:
            print("Please only enter a number.")
        else:
            return beta


def Falkner_Skan(eta, _f, beta):
    f = _f[0]
    fp = _f[1]
    fpp = _f[2]
    fppp = -(f*fpp + beta*(1 - fp**2))
    return fp, fpp, fppp


def optimize(fpp0, *args):
    sol = solve_ivp(Falkner_Skan,
                    t_span=(0, 10),
                    y0=(0, 0, fpp0),
                    t_eval=np.linspace(0, 10, 100),
                    args=args)
    return sol.y[1][-1] - 1


plt.figure()
betas = [-0.19884, -0.18, 0.0, 0.3]
for beta in betas:
    fpp0 = root_scalar(optimize, args=(beta,), x0=0, x1=10, bracket=(0, 5))
    print(fpp0)

    sol = solve_ivp(Falkner_Skan,
                    t_span=(0, 10),
                    y0=(0, 0, fpp0.root),
                    t_eval=np.linspace(0, 10, 100),
                    args=(beta,))
    plt.plot(sol.t, sol.y[1], label=f"$\\beta$={beta}")

plt.xlabel(r"$\eta$")
plt.ylabel("$f'$")
plt.title("Flakner-Skan")
plt.legend()
plt.show()
