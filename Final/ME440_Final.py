# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
ME440 Advanced Fluid Mechanics
Final Exam

Code for problem 3, turbulent Couette flow
"""
import numpy as np
# from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

plt.close("all")

# =============================================================================
# Functions
# =============================================================================
def Cf_implicit(Cf, Re, h, nu, kappa, B):
    """Implicit function for the friction coefficient found using the law of
    the wall and applying a symmetry boundary condition
    """
    return 8/(2*np.log(Re*np.sqrt(Cf/8))/kappa + 2*B)**2


def u_func(y, vstar, nu, h, kappa, B):
    """Top and bottom velocity functions using the law of the wall"""
    # Bottom
    if -h < y < 0:
        r = y + h
        return vstar*(np.log(vstar*r/nu)/kappa + B)
    # Top
    else:
        r = h - y
        return U - (vstar*(np.log(vstar*r/nu)/kappa + B))


def dudy_func(y, vstar, kappa, h):
    """Derivative of the velocity profile for the eddy viscosity calculation"""
    m = 1       # for changing the sign on y
    # Bottom
    if -h < y < 0:
        m = 1
    # Top
    else:
        m = -1
    return vstar/kappa/(h+m*y)


# =============================================================================
# Parameters
# =============================================================================
kappa = 0.41
B = 5.0
rho = 1.2           # kg/m^3; density of air
mu = 1.8e-5         # kg/m/s; dynamics viscosity of air
nu = mu/rho         # m^2/s; kinematic viscosity of air
h = 0.01            # m; half gap width
Re = 1e4            # Reynolds number
y = np.linspace(-h+1e-10, h-1e-10, 1000)

# =============================================================================
# Problem 3f
# =============================================================================
# Iteratively solving for the friction coefficient
Cf_old = 100
Cf = Cf_implicit(Cf_old, Re, h, nu, kappa, B)
rtol = 1e-6
while abs((Cf_old-Cf)/Cf_old) > rtol:
    Cf_old, Cf = Cf, Cf_implicit(Cf, Re, h, nu, kappa, B)
print(f"{Cf=}")

# Computing secondary values
U = Re*nu/h
tau_w = Cf*rho*U**2/8
vstar = np.sqrt(tau_w/rho)

plt.figure()
u = [u_func(_y, vstar, nu, h, kappa, B) for _y in y]

plt.plot(u, y, label="Turbulent")
plt.plot(U*(y/h+1)/2, y, label="Laminar")

plt.ylim(-h, h)
plt.xlim(0, U)

plt.xlabel(r"Mean Velocity $\bar{u}(y)$ [m/s]")
plt.ylabel(r"Position $y$ [m]")

plt.title("Position vs. Fluid Velocity, Couette Flow")
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("ME440_Final_P3f.png", dpi=200)

# =============================================================================
# Problem 3g
# =============================================================================
# Eddy viscosity calculation
dudy = np.array([dudy_func(_y, vstar, kappa, h) for _y in y])
mu_T = tau_w/dudy

plt.figure()
plt.plot(mu_T, y, label="Eddy Viscosity")
plt.axvline(mu, color="tab:orange", label="Molecular Viscosity")

plt.xticks(rotation=45)
plt.xlabel("Viscosity [kg/m/s]")
plt.ylabel("Position [m]")
plt.title("Position vs. Fluid Viscosity, Couette Flow")

plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("ME440_Final_P3g.png", dpi=200)
