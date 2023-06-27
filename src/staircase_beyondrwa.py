#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:45:15 2023

@author: jaya
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import definitions as sim
import scipy as scipy
from plotter_settings import *
from definitions import Osc_driven
from definitions import SysBath
import qutip as qt
import time
from tqdm import tqdm
def nb(w, T):
    kB = scipy.constants.Boltzmann
    hbar = scipy.constants.hbar
    return 1/(np.exp(hbar * w / (kB * T))  - 1)

def get_decay_rate(lindbladian):

    evals = lindbladian.eigenenergies() # diagonalizes 
    sorted_idx = np.argsort(-np.real(evals)) # sort according to negative real part
    evals = evals[sorted_idx]
    
    return  -np.real(evals[1])

#%%
def g3_giver(g4, K, wd):
    '''
    for a g3 and g4 get K
    '''
    g3 = np.sqrt(3 * wd / 20 * (3*g4/2 + K))
    
    return g3/3, g4/4, K, wd

def K_giver(g3, g4, wd):
    '''
    for a g3 and g4 get K
    '''
    K = 20*g3**2 / (3 * wd) - 3*g4/2
    
    return g3/3, g4/4, K, wd

def g4_giver(g3, K, wd):
    '''
    for a g3 and K get g4
    '''
    
    g4 = 2/3 * (20*g3**2 / (3 * wd) - K)
    return g3/3, g4/4, K, wd

#%%
# Computation for figure 1

# construct driven oscillator dictionary
e2 = 0
osc_paras = {'g3': 3*20*2*np.pi, 
              'g4': 4*0.280*2*np.pi, 
              'K': 0.320*2*np.pi, 
              'wd': 12000*2*np.pi, 
              'delta': 0, 
              'e2': e2}
e2_min = osc_paras['K']*0
e2_max = osc_paras['K']*20.2
e2_step = osc_paras['K']*0.2
e2s = np.arange(e2_min, e2_max, e2_step)
print (e2s/ osc_paras['K'], np.shape(e2s))

#%%
# set up sweep for params_varying_Kerr
max_order = 1
e2s_K = np.arange(0.0, 20.2, 0.2)
params = [osc_paras]
bare_decay_rates = np.zeros((len(params), len(e2s_K)))
for i, param in enumerate(tqdm(np.array(params), desc='parameter Progress')):
# for i, param in enumerate(np.array(params)):
    start = time.time()
    g3 = param['g3']#3*param['g3']*2*np.pi
    g4 = param['g4']#4*param[1]*2*np.pi
    K = param['K']#param[2]*2*np.pi
    wd = param['wd']#param[3]*2*np.pi
    
    e2s = K * e2s_K
    end0 = time.time()
    for j, e2 in enumerate(tqdm(np.array(e2s), desc='squeezing Progress')):
        print (e2/K, start - end0)
        
    
        osc_paras = {'g3': g3, 
                  'g4': g4, 
                  'K': K, 
                  'wd': wd, 
                  'delta': 0, 
                  'e2': e2}
    
        osc1 = Osc_driven(osc_paras)
        
        # perform the calculation to some order
        bare_kappa = osc1.K*0.025 #MHz (based on SNAILMON T1) 
        scale_kappa = {'-wd': 100, 'wd':100, '-K': 0, 'K':0, '-2wd':0, '2wd':0}
        bare_temp = 50e-3 
        scale_temp = {'-wd': 7, 'wd':7, '-K':1, 'K':1, '-2wd': 1, '2wd':1}
        
        sysbath = SysBath(osc1, osc_paras, max_order, bare_kappa, scale_kappa, bare_temp, scale_temp)
        
        c_ops = sysbath.get_c_ops_all(max_order)
                           
        H0 = osc1.get_hamiltonian_K()
        
        c_ops_rel = []
        for k in range(len(c_ops)):
            if c_ops[k] != qt.Qobj(np.zeros((sysbath.n_fock,sysbath.n_fock))):
                c_ops_rel.append(c_ops[k])
        
        lindbladian = qt.liouvillian(H0, c_ops_rel)
        bare_decay_rates[i, j] = get_decay_rate(lindbladian)
    end = time.time()
    print ('#####', start - end, param, '######')
#%%
fig, ax = plt.subplots()
ax.plot(e2s_K,  1/(osc_paras['K']*bare_decay_rates[i, :]*1e3), 'k')
ax.axhline(1e3, color = 'k', alpha = 0.5) # 1 sec
ax.axhline(1e0, color = 'k', alpha = 0.5) # 1 msec
# ax.axhline(36*1e8, color = 'k') # 1 hr
# ax.axhline(60*1e6, color = 'k') # 1 min

ax.set_ylim(1e-2, 1e6)
ax.set_xlim(0, 20)
ax.set_ylabel(r'$T_{X} ~(\mathrm{m s})$')
ax.semilogy()

#%%
# figure 1
path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure1'
title = r'/e2s_K'
np.savetxt(path_to_plot + title, e2s/K)


#%%
fig, ax = plt.subplots()

lifetime_mus = np.zeros((len(params), len(e2s_K)))

for i, param in enumerate(np.array(params[:])):
    g3 = param['g3']#3*param['g3']*2*np.pi
    g4 = param['g4']#4*param[1]*2*np.pi
    K = param['K']#param[2]*2*np.pi
    wd = param['wd']
    
    e2s = K * e2s_K

    for j, e2 in enumerate(e2s):
        
        osc_paras = {'g3': g3, 
                  'g4': g4, 
                  'K': K, 
                  'wd': wd, 
                  'delta': 0, 
                  'e2': e2}
    
        osc1 = Osc_driven(osc_paras)
        
        lifetime_mus[i, j] = 1/(osc_paras['K']*bare_decay_rates[i, j])
    title = '$in \mathrm{MHz}: g3/3/2 \pi = $' + str(np.round(param['g3']/3/2/np.pi, 2)) + '$, K/2\pi = $' + str(np.round(param['K']/2/np.pi, 2))
    if 1:
        ax.plot(e2s/osc_paras['K'], lifetime_mus[i, :], label = title)#, label = title_2)
    
    title = r'/'+str(max_order)+'_dominant_g3' + str(np.round(param['g3']/3/2/np.pi, 2)) + 'K' + str(np.round(param['K']/2/np.pi, 2))
    np.savetxt(path_to_plot + title, lifetime_mus[i, :])
    ax.legend()
    ax.set_xlabel(r'$\epsilon_2/K$')
    ax.set_ylabel(r'$T_{X} ~(\mathrm{m s})$')
    ax.semilogy()
#%%
#%%
# sensitivity to temperature: computation for figure 2
scale_factors = np.arange(1, 11, 1)
max_order = 1
# e2s_K = np.arange(0.0, 20.2, 0.2)
e2s_K = np.arange(0.0, 20.2, 0.2)
param = osc_paras
#%%
bare_decay_rates = np.zeros((len(scale_factors), len(e2s_K)))
for i, scale_factor in enumerate(tqdm(np.array(scale_factors), desc='temperature scaling Progress')):
# for i, param in enumerate(np.array(params)):
    start = time.time()
    g3 = param['g3']#3*param['g3']*2*np.pi
    g4 = param['g4']#4*param[1]*2*np.pi
    K = param['K']#param[2]*2*np.pi
    wd = param['wd']#param[3]*2*np.pi
    
    e2s = K * e2s_K
    end0 = time.time()
    for j, e2 in enumerate(tqdm(np.array(e2s), desc='squeezing Progress')):
        print (e2/K, start - end0)
        
    
        osc_paras = {'g3': g3, 
                  'g4': g4, 
                  'K': K, 
                  'wd': wd, 
                  'delta': 0, 
                  'e2': e2}
    
        osc1 = Osc_driven(osc_paras)
        
        # perform the calculation to some order
        bare_kappa = osc1.K*0.025 #MHz (based on SNAILMON T1) 
        scale_kappa = {'-wd': 100, 'wd':100, '-K': 0, 'K':0, '-2wd':0, '2wd':0}
        bare_temp = 50e-3 
        scale_temp = {'-wd': scale_factor, 'wd':scale_factor, '-K':1, 'K':1, '-2wd': 1, '2wd':1}
        
        sysbath = SysBath(osc1, osc_paras, max_order, bare_kappa, scale_kappa, bare_temp, scale_temp)
        
        c_ops = sysbath.get_c_ops_all(max_order)
                           
        H0 = osc1.get_hamiltonian_K()
        
        c_ops_rel = []
        for k in range(len(c_ops)):
            if c_ops[k] != qt.Qobj(np.zeros((sysbath.n_fock,sysbath.n_fock))):
                c_ops_rel.append(c_ops[k])
        
        lindbladian = qt.liouvillian(H0, c_ops_rel)
        bare_decay_rates[i, j] = get_decay_rate(lindbladian)
    end = time.time()
    print ('#####', start - end, param, '######')
#%%
# plot and save for figure 2
path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure2'
title = r'/e2s_K_2'
np.savetxt(path_to_plot + title, e2s_K)
#%%
lifetime_mus = np.zeros((len(scale_factors), len(e2s_K)))
fig, ax = plt.subplots()
base_temp = 50e-3
for i, scale_factor in enumerate(scale_factors):
    for j, e2 in enumerate(tqdm(np.array(e2s), desc='squeezing Progress')):
        lifetime_mus[i, j] = 1/(osc_paras['K']*bare_decay_rates[i, j])
    if 1:
        ax.plot(e2s/osc_paras['K'], lifetime_mus[i, :], label = str(base_temp*1e3*scale_factor) + 'mK')#, label = title_2)
        ax.legend()
    title = r'/'+str(max_order)+'_g3' + str(np.round(param['g3']/3/2/np.pi, 2)) + 'K' + str(np.round(param['K']/2/np.pi, 2)) + 'scale_temp' + str(scale_factor*base_temp)
    np.savetxt(path_to_plot + title, lifetime_mus[i, :])
    ax.semilogy()
#     
#%%
# computation for figure 3, sensitivity to K
Ks = np.array([0.25, 0.5, 1, 2, 4, 8])*2*np.pi
# Ks = np.array([0.32])*2*np.pi
g3 = 20 * 3 * 2 * np.pi
g4s = np.array([4*np.array(g4_giver(g3,K, 12000*2*np.pi)[1]) for K in Ks])
max_order = 2
e2s_K = np.arange(0.0, 20.2, 0.2)
bare_decay_rates = np.zeros((len(Ks), len(e2s_K)))
for i, K in enumerate(tqdm(np.array(Ks), desc='K variation Progress')):
    start = time.time()
    g3 = osc_paras['g3']#param['g3']#3*param['g3']*2*np.pi
    g4 = g4s[i]#param['g4']#4*param[1]*2*np.pi
    wd = osc_paras['wd']#param[3]*2*np.pi
    e2s = K * e2s_K
    end0 = time.time()
    for j, e2 in enumerate(tqdm(np.array(e2s), desc='squeezing Progress')):
        print (e2/K, start - end0)
        
    
        osc_paras = {'g3': 3*20*2*np.pi, 
                      'g4': g4s[i], 
                      'K': Ks[i], 
                      'wd': 12000*2*np.pi, 
                      'delta': 0, 
                      'e2': e2}
        print(osc_paras)
        osc1 = Osc_driven(osc_paras)
        
        # perform the calculation to some order
        bare_kappa = osc1.K*0.025 #MHz (based on SNAILMON T1) 
        scale_kappa = {'-wd': 100, 'wd':100, '-K': 1, 'K':1, '-2wd':1, '2wd':1}
        bare_temp = 50e-3 
        scale_temp = {'-wd': 7, 'wd':7, '-K':1, 'K':1, '-2wd': 1, '2wd':1}
        
        sysbath = SysBath(osc1, osc_paras, max_order, bare_kappa, scale_kappa, bare_temp, scale_temp)
        
        c_ops = sysbath.get_c_ops_all(max_order)
                           
        H0 = osc1.get_hamiltonian_K()
        
        c_ops_rel = []
        for k in range(len(c_ops)):
            if c_ops[k] != qt.Qobj(np.zeros((sysbath.n_fock,sysbath.n_fock))):
                c_ops_rel.append(c_ops[k])
        
        lindbladian = qt.liouvillian(H0, c_ops_rel)
        bare_decay_rates[i, j] = get_decay_rate(lindbladian)
    end = time.time()
    print ('#####', start - end, K, '######')

#%%
# figure 3
path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure3'
title = r'/e2s_K_2'
#%%
# plot and save for figure 3
lifetime_mus = np.zeros((len(Ks), len(e2s_K)))
fig, ax = plt.subplots()
base_temp = 50e-3
for i, K in enumerate(tqdm(np.array(Ks), desc='K variation Progress')):
    for j, e2 in enumerate(tqdm(np.array(e2s), desc='squeezing Progress')):
        lifetime_mus[i, j] = 1/(K*bare_decay_rates[i, j])
    if 1:
        ax.plot(e2s/osc_paras['K'], lifetime_mus[i, :]/1e3, label = str(K/2/np.pi) + 'MHz')#, label = title_2)
        ax.legend()
    title = r'/'+str(max_order)+'_g3' + str(np.round(g3/3/2/np.pi, 2)) + 'K' + str(np.round(K/2/np.pi, 2))
    np.savetxt(path_to_plot + title, lifetime_mus[i, :])
    ax.semilogy()
#%%
# computation for figure 5: with 2-photon cooling
# adding two photon dissipation to the full effective Lindbladian
if 1:
    
    max_orders = np.array([1, 2])
    
    e2s_K = np.arange(0.0, 20.2, 0.2)
    bare_decay_rates = np.zeros((len(e2s_K), len(max_orders)))
    for i, e2_K in enumerate(tqdm(np.array(e2s_K), desc='squeezing Progress')):
        c_ops_rel = []
        e2 = e2_K * 0.320*2*np.pi
        osc_paras = {'g3': 3*20*2*np.pi, 
                      'g4': 4*0.280*2*np.pi, 
                      'K': 0.320*2*np.pi, 
                      'wd': 12000*2*np.pi, 
                      'delta': 0, 
                      'e2': e2}
        
        osc1 = Osc_driven(osc_paras)
        
     
        
        for j, max_order in enumerate(max_orders):
            # perform the calculation to some order
            bare_kappa = osc1.K*0.025 #MHz (based on SNAILMON T1) 
            scale_kappa = {'-wd': 100, 'wd':100, '-K': 1, 'K':1, '-2wd':1, '2wd':1}
            bare_temp = 50e-3 
            scale_temp = {'-wd': 7, 'wd':7, '-K':1, 'K':1, '-2wd': 1, '2wd':1}
            
            sysbath = SysBath(osc1, osc_paras, max_order, bare_kappa, scale_kappa, bare_temp, scale_temp)
            
            c_ops = sysbath.get_c_ops_all(max_order)
                               
            H0 = osc1.get_hamiltonian_K()
            
            if j == 2:
                print ('relevant prefactors:', e2/sysbath.K)
                print ('order 0, wd/2:', sysbath.kappas['wd/2'] * sysbath.nbars['wd/2'])
                print ('order 1, wd:', sysbath.kappas['wd'] * sysbath.nbars['wd'] * (8*sysbath.g3/3/sysbath.wd )**2)
                print ('order 2, K:', sysbath.kappas['K'] * sysbath.nbars['K'] * (32*sysbath.g3**2*sysbath.Pi/sysbath.wd**2 )**2)
                print ('order 2, 2wd:', sysbath.kappas['2wd'] * sysbath.nbars['2wd'] * (224*sysbath.g3**2*sysbath.Pi/(45*sysbath.wd**2) + 16/5*sysbath.g4*sysbath.Pi/sysbath.wd)**2)
                     
            H0 = osc1.get_hamiltonian_K()
        
            c_ops_rel = []
            for k in range(len(c_ops)):
                if c_ops[k] != qt.Qobj(np.zeros((sysbath.n_fock,sysbath.n_fock))):
                    c_ops_rel.append(c_ops[k])
                    
            # adding two photon dissipation
            prefactor = 0.01
            c_ops_rel.append(np.sqrt(prefactor) * sysbath.sys.a**2)
            print ('add cooling np.sqrt(prefactor) (kHz):', 1e3*prefactor)
        
            lindbladian = qt.liouvillian(H0, c_ops_rel)
            bare_decay_rates[i, j] = get_decay_rate(lindbladian)
#%%
# plot and save for figure 2
path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure4'
title = r'/e2s_K_2'
np.savetxt(path_to_plot + title, e2s_K)
#%%
lifetime_mus = np.zeros((len(e2s_K), len(max_orders)))
fig, ax = plt.subplots()
color = ['red', 'blue', 'green']
base_temp = 50e-3
for j, e2 in enumerate(tqdm(np.array(e2s_K), desc='squeezing Progress')):
    for i, max_order in enumerate(tqdm(np.array(max_orders), desc='orders Progress')):
        lifetime_mus[j, i] = 1/(osc_paras['K']*bare_decay_rates[j, i])
# ax.plot(np.arange(0.0, 20.2, 0.2)[::10], lifetimes_0_f/1e3, label = r'order 0 no cooling')
# ax.plot(np.arange(0.0, 20.2, 0.2)[::10], lifetimes_1_f/1e3, label = r'order 1 no cooling')
# ax.plot(np.arange(0.0, 20.2, 0.2)[::10], lifetimes_2_f/1e3, label = r'order 2 no cooling')
for i, max_order in enumerate(tqdm(np.array(max_orders), desc='orders Progress')):
    ax.plot(e2s_K, lifetime_mus[ :, i]/1e3, label = str(max_order) + 'order, w/cooling', color = color[i], dashes = (4, i+1))#, label = title_2)
    ax.legend()
    title = r'/'+str(max_order)+ 'cooling'
    
    ax.semilogy()
    np.savetxt(path_to_plot + title, lifetime_mus[:, i])
