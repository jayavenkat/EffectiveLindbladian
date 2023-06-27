#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:08:50 2023

@author: jaya
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
#from qutip import basis, sigmaz, sigmap, sigmam, mesolve, Options
from scipy.constants import hbar
from definitions_sko import *
from matplotlib import cm
from tqdm import tqdm
import time
#%%

# path to save plots
save_path = r'/Users/jaya/My Drive/PhD Work/Papers/Iachello/lindbladian-symmetries/data/notes/lindbladian-symmetries/figures'
# set_plot_settings('paper')

# time domain simulations
# system parameters
n_fock = 10
omega = 0          # Oscillator frequency (arbitrary units)
kerr = 1            # Kerr nonlinearity
squeezing = 3     # squeezing strength
n_fock = 10
n_fock = (1*(np.ceil(squeezing**2))).astype(int)
if n_fock < 40:
    n_fock = 40
# n_fock = 10
timesteps = 1000               # Number of time steps for the simulation
t_final = 2000                   # Final time for the simulation (arbitrary units)

# bath parameters
kappa = 0.1                   # Damping rate (arbitrary units)
n_th = 0.0 #* 0                  # thermal photon occupation number
kappa_loss = kappa * (1 + n_th) 
kappa_gain = kappa * n_th

# Define the Hamiltonian, collapse operator and initial state
a = destroy(n_fock)            # Annihilation operator
a_dag = a.dag()
h =  -omega * a_dag * a   + kerr * a_dag**2 * a**2 \
    -squeezing * (a_dag**2) - np.conjugate(squeezing) * a**2   # Hamiltonian
c_ops = [np.sqrt(kappa_loss) * a, np.sqrt(kappa_gain) * a_dag]
psi0 = basis(n_fock, 2)        # Initial state (5 quanta)

# Time vector for the simulation
# t = np.linspace(0, t_final, timesteps)

# # Options for the solver
# opts = Options(store_states=True, nsteps=5000)

# # Solve the master equation
# result = mesolve(h, psi0, t, c_ops, [a_dag*a], options=opts)

# Plot the results
# plt.close('all')
# fig, ax = plt.subplots()
# ax.plot(kappa*t, result.expect[0], label = r'numerical$, n_{\mathrm{th}} = $' + str(n_th))
# ax.plot(kappa*t, result.expect[0][0]*np.exp(-kappa*t), linestyle = 'dashed', \
#         dashes = (2, 2), label = r'analytical$, \langle n \rangle = \langle n_0 \rangle e^{- \kappa t} $')
# #ax.plot(kappa*t, result.expect[0][0]*np.exp(-kappa*t), 'k', label = 'time-domain analytical')
# ax.semilogy()
# plt.legend()
# plt.xlabel(r'$\kappa t$')
# plt.ylabel('Occupation number')
# xticks = ax.get_xticks()
# ax.set_xticklabels(replace_minus_with_hyphen(xticks))
# yticks = ax.get_yticks()
# ax.set_yticklabels(replace_minus_with_hyphen(yticks))
# plt.title('Decay of a damped squeezed kerr oscillator')
# plt.show()
# set_figure_size(3, 2, ax)

# title = r'/average_photon_number_DSKO'+str(n_th)+'.png'
# plt.savefig(save_path + title, dpi = 900, bbox_inches = 'tight')

#%%

# diagonalize the Lindbladian
start = time.time()

l = get_lindbladian(n_fock,omega, kerr, squeezing, kappa_loss, kappa_gain)
l_eigs = (l.eigenenergies()[::-1])
fig, ax = plt.subplots()

ax.scatter(np.real(l_eigs), np.imag(l_eigs), color =  'k', facecolor = 'none', label = 'numerical diagonalization')
#Identify degeneracies in the real part
real_part = np.real(l_eigs)
unique_real, counts_real = np.unique(real_part, return_counts=True)
degenerate_real = unique_real[counts_real > 1]

# Identify degeneracies in the imaginary part
imag_part = np.imag(l_eigs)
unique_imag, counts_imag = np.unique(imag_part, return_counts=True)
degenerate_imag = unique_imag[counts_imag > 1]

end = time.time()

print(start - end)
# # Plot degenerate points with different marker color or size
# for eig in l_eigs:
#     if np.real(eig) in degenerate_real or np.imag(eig) in degenerate_imag:
#         ax.scatter(np.real(eig), np.imag(eig), color='red', facecolor='none', marker='o')


# for j in range(n_fock):
#     for k in range(n_fock):
#         ax.scatter(-kappa_loss/2* (j+k), (j-k)*( omega   - kerr* (j+k - 1)), s = 2, facecolor = 'red'  , label = 'analytical (no squeezing)' if j == 0 and k ==0 else "")
ax.set_xlabel(r'$\mathrm{Re}(\mathrm{eigs}(L))$')
ax.set_ylabel(r'$\mathrm{Im}(\mathrm{eigs}(L))$')
ax.legend()
title = r'SKO Lindbladian with $\bar{n}_{\mathrm{th}} = $' + str(n_th) + ', $\omega$ = ' + str(omega) + '$, K$ = ' + str(kerr) +'$, \epsilon_2$ = ' + str(squeezing) + ', $\kappa = $'+str(kappa) 
plt.suptitle(title)

# xticks = list(np.linspace(-0.7, 0., 8))
# ax.set_xticks(xticks)
# yticks = list(np.arange(-6, 6+1, 2))
# ax.set_yticks(yticks)

# ax.set_xticklabels(replace_minus_with_hyphen(np.round(xticks, 1)))
# ax.set_yticklabels(replace_minus_with_hyphen(np.round(yticks, 1)))
#save_path = r'/Users/jaya/My Drive/PhD Work/Papers/Iachello/lindbladian-symmetries/data/notes/lindbladian-symmetries/figures'
title = r'/SKO_spectrum'  + str(omega) + str(kerr) + str(squeezing) + str(n_th)+'_10.png'
plt.savefig(save_path + title, dpi = 900, bbox_inches = 'tight')
# set_plot_settings('paper')
set_figure_size(5, 3, ax)

#%%
# Plot TX vs squeezing
squeezings = np.arange(0, 9.7, 0.1)
# squeezings = np.arange(1.4, 1.5, 0.01)
n_focks = (5*(np.ceil(squeezings))).astype(int)
n_focks = (np.ones(len(squeezings))*(5*np.ceil(np.max(squeezings)))).astype(int)
n_focks = (np.ones(len(squeezings))*(1.2*np.ceil((squeezings**2)))).astype(int)
n_focks[n_focks < 18] = 18
# n_focks = (np.ones(len(squeezings))*10).astype(int)
smallest_real = np.zeros(len(squeezings))
smallest_imag = np.zeros(len(squeezings))
start = time.time()
for i, squeezing in enumerate(tqdm(squeezings, desc='Squeezing Progress')):
    l = get_lindbladian(n_focks[i],omega, kerr, squeezing, kappa_loss, kappa_gain)
    l_eigs = (l.eigenenergies()[::-1])
    smallest = find_closest_eigenvalue(l_eigs)
    smallest_real[i], smallest_imag[i] = np.real(smallest), np.imag(smallest)
    # Simulate the task with sleep
    print(squeezing)
    time.sleep(1)  # Replace this with your actual task
end = time.time()
print('####')
print (np.abs(end-start))
print('####')
#%%
# fig, ax = plt.subplots(2, 1)
# for i in range(9):
#     ax[0].plot(squeezings, (sorted_l_eigs_all_re[:, i]), 'k.')
#     ax[1].plot(squeezings, (sorted_l_eigs_all_im[:, i]), 'k.')
#     # if i == 1:
#     #     ax[0].plot(squeezings, (sorted_l_eigs_all_re[:, -i]), 'r.')
#     #     ax[1].plot(squeezings, (sorted_l_eigs_all_im[:, -i]), 'r.')
#     # if i == 1:
#     #     axs[0].plot(squeezings, l_eigs_all_re[:, i], 'r', '.')
#     #     axs[1].plot(squeezings, l_eigs_all_im[:, i], 'r', '.')
# # yticks0 = list(np.linspace(-0.2, 0.2, 2))
# # yticks1 = list(np.linspace(-10.2, 10.2, 2))
# # ax.set_xticks(xticks)
# # ax[0].set_yticks(yticks0)
# # ax[1].set_yticks(yticks1)
# title = r'SKO Lindbladian with $\bar{n}_{\mathrm{th}} = $' + str(n_th) + ', $\omega$ = ' + str(omega) + '$, K$ = ' + str(kerr) + ', $\kappa = $'+str(kappa) 
# plt.suptitle(title)
# # axs[1].set_ylim(-10.5, 10.5)
# title = r'/SKO_re_im_eigs'  + str(omega) + str(kerr) + str(max(squeezings))+str(max(n_focks))+'.png'
# plt.savefig(save_path + title, dpi = 900, bbox_inches = 'tight')
#%%
fig, ax = plt.subplots(2, 1)
ax[0].plot(squeezings, smallest_real, 'k.')
# ax[0].axvline(2, label = r'$\epsilon_2/K = 2$', color = 'k', linestyle = '--')
# ax[0].axvline(3.9, label = r'$\epsilon_2/K = 3.9$', color = 'k', linestyle = '--')
ax[0].legend()
ax[1].plot(squeezings, smallest_imag, 'k.')
# ax[1].set_ylim(-1.2,1.2)
ax[0].set_xlabel(r'$\epsilon_2/K$')
ax[0].set_ylabel(r'Real part')
ax[1].set_ylabel(r'imaginary part')
ax[0].set_xlim(0, 10)
ax[1].set_xlim(0, 10)
ax[1].set_ylim(-1.2, 1.2)
title = r'SKO Lindbladian with $\bar{n}_{\mathrm{th}} = $' + str(n_th) + ', $\omega$ = ' + str(omega) + '$, K$ = ' + str(kerr) + ', $\kappa = $'+str(kappa) 
plt.suptitle(title)

# set_size(4,2, ax[0])
# ax.semilogy()
title = r'/eigs'  + str(omega) + str(kerr) + str(squeezing)+'.png'
plt.savefig(save_path + title, dpi = 900, bbox_inches = 'tight')
#%%
fig, ax = plt.subplots()
ax.plot(squeezings, -1/smallest_real, 'k.')
ax.semilogy()
title = r'$T_X$ with $\bar{n}_{\mathrm{th}} = $' + str(n_th) + ', $\omega$ = ' + str(omega) + '$, K$ = ' + str(kerr) + ', $\kappa = $'+str(kappa) 
plt.suptitle(title)
ax.set_xlabel(r'$\epsilon_2/K$')
ax.set_ylabel(r'$K T_X$')
ax.set_ylim(1, 1e10)
ax.set_xlim(0, 9)
title = r'/TX'  + str(omega) + str(kerr) + str(squeezing)+'.png'
plt.savefig(save_path + title, dpi = 900, bbox_inches = 'tight')
