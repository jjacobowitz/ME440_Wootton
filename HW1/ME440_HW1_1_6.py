# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
ME440

Homework 1 Problem 1-6
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

def velocity(r, t):
    C = nu = 1
    v_t = C/r*(1 - np.exp(-r**2/(4*nu*t)))
    return v_t

def vorticity(r, t):
    C = nu = 1
    w_z = C/(2*nu*t)*np.exp(-r**2/(4*nu*t))
    return w_z

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

r = np.linspace(0, 10, 1000)
t = [0.000001, 0.5, 1., 2., 4., 8., 16., 32., 64., 128.]

for _t in t:
    ax1.plot(r, velocity(r, _t), label=f"t={_t:.1f}")
    ax2.plot(r, vorticity(r, _t), label=f"t={_t:.1f}")
    
ax1.set_xlabel(r"$r$")
ax1.set_ylabel(r"$v_\theta$")
ax1.set_title(r"$v_\theta$ vs $r$ at Various Times ($t$)")
ax1.set_ylim(0, 1)
ax1.legend()

ax2.set_xlabel("$r$")
ax2.set_ylabel("$\omega_z$")
ax2.set_title("$\omega_z$ vs $r$ at Various Times ($t$)")
ax2.set_ylim(0, 2)
ax2.legend()

fig.show()
fig.savefig("ME440_HW1_1_6.png", dpi=300)