#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:12:25 2023

@author: jaya
"""

# plotter
from plotter_settings import *
import time
from tqdm import tqdm
from seaborn import *
#%%
set_plot_settings('paper')
'''
Initialize Colors
'''
zero = color_palette("RdBu", 10)[0]
#%%
zero = color_palette("Paired", 10)[7]
zero =  (24./255., 157./255., 194./255)
one = 'darkorange'
two = color_palette("RdBu", 10)[0]
three = 'green'
four = (93./255, 136./255, 54./255)
five = (141/255., 52/255., 120/255.)
six = (236/255., 225/255., 51/255.)

seven = 'palevioletred'#(220/255., 148/255., 162/255.)
eight = (169/255., 198/255., 105/255.)
nine = (166/255., 156/255., 199/255.)
ten = color_palette("YlOrRd", 10)[2]
eleven = (142/255., 194./255, 198./255) 
twelve = 'plum'#sns.color_palette("PiYG", 10)[3]
thirteen = color_palette("RdYlGn", 10)[-5]
fourteen = color_palette("RdYlBu", 10)[-4]
fifteen = color_palette("RdPu", 10)[2]
sixteen = light_palette("navy", reverse=True)[-1]
arbit_high= color_palette("Set2")[-2]
arbit_higher = 'lightgray'
'''
Potentials
'''
LeatherJacket = '#708090'
BlackBean = '#32174D'#(61/255., 12/255., 2/255.)

states = [
        zero,
        one,
        two,
        three,
        four,
        five,
        six,
        seven,
        eight,
        nine,
        ten,
        eleven,
        twelve,
        thirteen,
        fourteen,
        fifteen,
        sixteen,
        arbit_high,
        arbit_higher
        ]
states_all = states + [states[-1]]*100 
#%%

# figure 1

path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure1'
# load data
e2s_K = np.loadtxt(path_to_plot + r'/e2s_K')
lifetimes_0_f = np.loadtxt(path_to_plot + r'/0g320.0K0.32')
lifetimes_1_f = np.loadtxt(path_to_plot + r'/1g320.0K0.32')
lifetimes_2_f = np.loadtxt(path_to_plot + r'/2g320.0K0.32')
lifetimes_1_m = np.loadtxt(path_to_plot + r'/1_dominant_g320.0K0.32')
#%%
set_plot_settings('paper')
fig, ax = plt.subplots()
ax.axhline(1e3, color = 'k', alpha = 0.5) # 1 sec
ax.axhline(1e0, color = 'k', alpha = 0.5) # 1 msec
# ax.axhline(36*1e8, color = 'k') # 1 hr
# ax.axhline(60*1e6, color = 'k') # 1 min
ax.plot(e2s_K, lifetimes_0_f/1e3, label = r'order 0: full Lindbladian', lw =2, color = states[0])
ax.plot(e2s_K, lifetimes_1_f/1e3, label = r'order 1: full Lindbladian', lw =2, color = states[1])
ax.plot(e2s_K, lifetimes_1_m/1e3, label = r'order 1: main terms', dashes = (2,1.5), lw =2, color =  'green')
ax.plot(e2s_K, lifetimes_2_f/1e3, label = r'order 2: full Lindbladian', dashes = (4, 3), lw =2, color = states[2])

ax.legend()
ax.set_xlabel(r'$\epsilon_2/K = |\alpha|^2$')
ax.set_ylabel(r'$T_{X} ~(\mathrm{m s})$')
ax.semilogy()

ax.set_xlim(-0.1, 20.1)
ax.set_ylim(min(lifetimes_0_f/1e3), max(lifetimes_0_f/1e3))
set_size(2.95, 2.1, ax)
path_to_save = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/figures/figure1'

plt.savefig(path_to_save + r'/fig1.pdf', dpi = 900, transparent = True, bbox_inches = 'tight')
#%%

# figure 2

path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure2'
scale_factors = np.arange(1, 11, 1)
lifetime_mus = np.zeros((len(scale_factors), len(e2s_K)))
max_order = 1
g3 = 20.0
K = 0.32
fig, ax = plt.subplots()
base_temp = 50e-3
# load data
ax.plot(e2s_K, lifetimes_0_f/1e3, label = r'order 0', lw =2,  color = states[0])
e2s_K = np.loadtxt(path_to_plot + r'/e2s_K_2')
for i, scale_factor in enumerate(scale_factors[:]):
    title = r'/'+str(max_order)+'_g3' + str(np.round(g3, 2)) + 'K' + str(np.round(K, 2)) + 'scale_temp' + str(scale_factor*base_temp)
    lifetime_mus[i, :] = np.loadtxt(path_to_plot + title)
    ax.plot(e2s_K, (lifetime_mus[i, :]/1e3), label = str(int(base_temp*1e3*scale_factor)) + ' mK', lw = 2, color = states[i+5])

ax.axhline(1e0, color = 'k', alpha = 0.5) # 1 msec
ax.axhline(1e3, color = 'k', alpha = 0.5) # 1 sec
# ax.plot(e2s_K, lifetimes_1_f/1e3, label = r'order 1: old', lw =2, dashes = (2,2), linestyle = 'dashed', color = states[1])
ax.legend(ncol = 3)

ax.set_xlim(-0.1, 20.1)
ax.set_ylim(min(lifetimes_0_f/1e3), max(lifetimes_0_f/1e3))
set_size(2.85, 2.2, ax)   
ax.set_xlabel(r'$\epsilon_2/K = |\alpha|^2$')
ax.set_ylabel(r'$T_{X} ~(\mathrm{m s})$')
ax.semilogy()
path_to_save = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/figures/figure2'

plt.savefig(path_to_save + r'/fig2.pdf', dpi = 900, transparent = True, bbox_inches = 'tight')
# ax.set_xlim(-0.1, 20.1)
# ax.set_ylim(-0.1, 1e5)
#%%
# figure 3

path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure3'
#%%
states = [
        zero,
        eight,
        four,
        seven,
        
        
        one,
        two,
        six,
        
        seven,
        eight,
        nine,
        ten,
        eleven,
        twelve,
        thirteen,
        fourteen,
        fifteen,
        sixteen,
        arbit_high,
        arbit_higher
        ]
palplot(states)
def g4_giver(g3, K, wd):
    '''
    for a g3 and K get g4
    '''
    
    g4 = 2/3 * (20*g3**2 / (3 * wd) - K)
    return g3/3, g4/4, K, wd
# figure 3
Ks = np.array([0.25, 0.5, 1, 2, 4, 8])*2*np.pi
g3 = 20 * 3 * 2 * np.pi
g4s = np.array([4*np.array(g4_giver(g3,K, 12000*2*np.pi)[1]) for K in Ks])
max_order = 2
e2s_K = np.loadtxt(path_to_plot + r'/e2s_K_2')
bare_decay_rates = np.zeros((len(Ks), len(e2s_K)))
lifetime_mus = np.zeros((len(Ks), len(e2s_K)))
fig, ax = plt.subplots()
for i, K in enumerate(tqdm(np.array(Ks), desc='K variation Progress')):
    g4 = g4s[i]#param['g4']#4*param[1]*2*np.pi
    e2s = K * e2s_K
    title = r'/'+str(max_order)+'_g3' + str(np.round(g3/3/2/np.pi, 2)) + 'K' + str(np.round(K/2/np.pi, 2))
    lifetime_mus[i, :] = np.loadtxt(path_to_plot + title)
    ax.plot(e2s_K, (lifetime_mus[i, :]/1e3), label = str(np.round(K/2/np.pi, 2))+ ' MHz', color = states[i])

ax.legend(ncol = 2)
ax.axhline(1e0, color = 'k', alpha = 0.5) # 1 msec

ax.set_xlabel(r'$\epsilon_2/K = |\alpha|^2$')
ax.set_ylabel(r'$T_{X} ~(\mathrm{m s})$')
ax.set_xlim(-0.1, 20.1)
ax.set_ylim(min(0.9*lifetime_mus[-1, :]/1e3), max(1.01*lifetime_mus[0, :]/1e3))
set_size(2.85, 2.1, ax)       
ax.semilogy()
path_to_save = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/figures/figure3'

plt.savefig(path_to_save + r'/fig3.pdf', dpi = 900, transparent = True, bbox_inches = 'tight')
#%%
# figure 4, compare with experimental data
path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure3/experiment'
title = r'/e2_K'
e2_K_exp =  np.loadtxt(path_to_plot + title)
title = r'/lifetimes'
lifetime_exp = np.loadtxt(path_to_plot + title)
states = [
        zero,
        one,
        two,
        three,
        four,
        five,
        six,
        seven,
        eight,
        nine,
        ten,
        eleven,
        twelve,
        thirteen,
        fourteen,
        fifteen,
        sixteen,
        arbit_high,
        arbit_higher
        ]
# reload from figure 1
fig, ax = plt.subplots()
# figure 1

path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure1'
# load data
e2s_K = np.loadtxt(path_to_plot + r'/e2s_K')
lifetimes_0_f = np.loadtxt(path_to_plot + r'/0g320.0K0.32')
lifetimes_1_f = np.loadtxt(path_to_plot + r'/1g320.0K0.32')
lifetimes_2_f = np.loadtxt(path_to_plot + r'/2g320.0K0.32')
lifetimes_1_m = np.loadtxt(path_to_plot + r'/1_dominant_g320.0K0.32')
ax.axhline(1e0, color = 'k', alpha = 0.5) # 1 msec
ax.plot(e2_K_exp, lifetime_exp/1e3, 'o', color = 'black', markersize = 6,  mfc = 'none', label = r'experimental data')
ax.plot(e2s_K, lifetimes_0_f/1e3, label = r'order 0: full', lw =2, color = states[0])
ax.plot(e2s_K, lifetimes_1_f/1e3, label = r'order 1: full', lw =2, color = states[1])
ax.plot(e2s_K, lifetimes_1_m/1e3, label = r'order 1: main', dashes = (2,1.5), lw =2, color =  'green')
ax.plot(e2s_K, lifetimes_2_f/1e3, label = r'order 2: full', dashes = (4, 3), lw =2, color = states[2])
ax.semilogy()
ax.set_xticks(list(np.arange(0, 11, 2)))
ax.set_xlim(0, 10)
ax.set_ylim(0.008, lifetimes_0_f[52]/1e3)
# set_plot_settings('paper')
plt.legend(loc = 'upper left', ncol = 1)#, transparent  = False)
ax.set_xlabel(r'$\epsilon_2/K$')
ax.set_ylabel(r'$T_X (\mathrm{ms})$')
set_size(2.85, 2.2, ax)       
ax.semilogy()
path_to_save = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/figures/figure4'

plt.savefig(path_to_save + r'/fig4.pdf', dpi = 900)#, transparent = True, bbox_inches = 'tight')
#%%
# figure 5 w/ cooling
max_orders = np.array([0, 1, 2])
# no cooling data 
path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure1'
# load data
e2s_K_nc = np.loadtxt(path_to_plot + r'/e2s_K')
lifetimes_0_f_nc = np.loadtxt(path_to_plot + r'/0g320.0K0.32')
lifetimes_1_f_nc = np.loadtxt(path_to_plot + r'/1g320.0K0.32')
lifetimes_2_f_nc = np.loadtxt(path_to_plot + r'/2g320.0K0.32')
# w/ cooling data 
path_to_plot = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/data/figure4'
e2s_K_c = np.loadtxt(path_to_plot + r'/e2s_K_2')
lifetimes_0_f_c = np.loadtxt(path_to_plot + r'/0cooling')
lifetimes_1_f_c = np.loadtxt(path_to_plot + r'/1cooling')
lifetimes_2_f_c = np.loadtxt(path_to_plot + r'/2cooling')
fig, ax = plt.subplots()

ax.plot(e2s_K_nc, lifetimes_0_f_nc/1e3, label = r'order 0: full', lw =2, color = states[0])
# ax.plot(e2s_K_nc, lifetimes_1_f_nc/1e3, label = r'order 1: full Lindbladian', lw =2, color = states[1])
ax.plot(e2s_K_nc, lifetimes_2_f_nc/1e3, label = r'order 2: full', lw =2, color = states[2])
# ax.plot(e2s_K_c, lifetimes_0_f_c/1e3, label = r'order 0: cooling', dashes = (1,1), lw =2, color = 'k')
# ax.plot(e2s_K_c, lifetimes_1_f_c/1e3, label = r'order 1: cooling', lw =2, color = 'r')
ax.plot(e2s_K_c, lifetimes_2_f_c/1e3, label = r'order 2: w/ 2-photon cooling', lw =2, color = 'b')
ax.axhline(1e3, color = 'k', alpha = 0.5) # 1 sec
ax.axhline(1e0, color = 'k', alpha = 0.5) # 1 msec
ax.set_xticks(list(np.arange(0, 21, 2)))
ax.set_xlim(-0.1, 20.1)
ax.set_ylim(min(lifetimes_0_f_nc/1e3), max(lifetimes_0_f/1e3))
set_size(2.85, 2.1, ax)
ax.legend()
ax.set_xlabel(r'$\epsilon_2/K = |\alpha|^2$')
ax.set_ylabel(r'$T_{X} ~(\mathrm{m s})$')
ax.semilogy()
path_to_save = r'/Users/jaya/My Drive/PhD Work/Papers/effective_Lindbladian/effective_Lindbladian_code/EffectiveLindbladian/figures/figure5'

plt.savefig(path_to_save + r'/fig5.pdf', dpi = 900, transparent = True, bbox_inches = 'tight')
# ax.set_ylim(0.008, 1e3)
