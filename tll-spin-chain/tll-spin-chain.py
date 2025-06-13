import os
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh
import numpy as np
import scipy as sp
import jax
import time

'''
Hxy = - (hbar * J / 2) SUM(i<j) 1/(r^3 of i-j) (Sx_i Sx_j + Sy_i Sy_j)
J = 2pi * 0.55 MHz
'''

J = 2 * sp.constants.pi * 0.55e6  # J
V = -J * sp.constants.hbar / 2

start_time = time.time()

N = 24
hi = nk.hilbert.Spin(s=1 / 2, N=N)

Gamma = -1

H = sum([V * sigmaz(hi, i) * sigmaz(hi, (i + 1) % N) for i in range(N)])