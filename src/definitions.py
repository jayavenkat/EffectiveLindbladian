#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:43:04 2023

@author: jaya
"""

import numpy as np
from scipy.optimize import curve_fit
import qutip as qt
import scipy as scipy

def nb(w, T):

    kB = scipy.constants.Boltzmann
    hbar = scipy.constants.hbar
    return 1/(np.exp(hbar * w / (kB * T))  - 1)

class Osc_driven:
    def __init__(self, osc_dict):
        self.g3 = osc_dict['g3']
        self.g4 = osc_dict['g4']
        self.K = osc_dict['K']
        self.e2 = osc_dict['e2']
        self.Pi = osc_dict['e2']/osc_dict['g3']
        self.wd = osc_dict['wd']
        self.delta = osc_dict['delta']
        self.n_fock = self.get_n_fock()
        self.N_max = self.get_N_max()
        self.change_basis()
        
    def get_n_fock(self):
        K = self.K
        e2 = self.e2
        delta = self.delta
        if e2/K + delta/K/2 > 0: #check?
            a_eq = np.sqrt(e2/K + delta/K/2)
        else:
            a_eq = 0
    
        # coarsely determines truncation based on classical eqb point
        n_fock = max(int(np.round(5.3 * np.abs(a_eq)**2)), 16)
        
#         print(n_fock)
        return max(200, n_fock)

    def get_N_max(self):
        K = self.K
        e2 = self.e2
        delta = self.delta
        return 50
        
        
    def get_hamiltonian_K_Fock(self):
        wd = self.wd
        e2 = self.e2
        g3 = self.g3
        g4 = self.g4
        Pi = self.Pi
        K = self.K
        delta = self.delta
        a = qt.destroy(self.n_fock)
        adag = a.dag()
        H0_K = delta/ K * adag * a - adag * adag * a * a + e2/K * adag * adag + np.conjugate(e2/K) * a * a
        return H0_K
    
    def get_hamiltonian_K(self):
#         H0_K_Fock = self.get_hamiltonian_K_Fock()
        eig_energies, eig_states = self.get_eigenstates()
        H0_K = qt.Qobj(np.column_stack(list(qt.basis(self.n_fock, i)*eig_energies[i] \
                                                     for i in range(len(eig_states)))) \
                                     )
#         self.hamiltonian_K = cbm.dag() * self.get_hamiltonian_K_Fock() * cbm
        self.H0_K = qt.Qobj(H0_K[-self.N_max:, -self.N_max:])
        return self.H0_K
    
    def get_eigenstates(self):
        eig_energies, eig_states = self.get_hamiltonian_K_Fock().eigenstates()
        return eig_energies, eig_states
    
    def change_basis(self):
        eig_energies, eig_states = self.get_eigenstates()
        cbm = qt.Qobj(np.column_stack(list(x.full() \
                                                     for x in eig_states)) \
                                     )
        # Changing basis to qubit eigenstates
        self.a = cbm.dag() * qt.destroy(self.n_fock) * cbm
        self.a = qt.Qobj(self.a[-self.N_max:, -self.N_max:])
        
        
        
        
class SysBath():
    def __init__(self, sys, osc_dict, max_order, bare_kappa, scale_kappa, bare_temp, scale_temp):
        self.sys = sys
        self.freqs = self.get_freqs()
        self.kappas = self.get_kappas(bare_kappa, scale_kappa)
        self.temps = self.get_temps(bare_temp, scale_temp)
        self.nbars = self.get_nbars(True)
        
    
    def get_freqs(self):
        wd = self.sys.wd
        
        return {'-5wd/2':-5*wd/2, '-2wd':-4*wd/2, '-3wd/2':-3*wd/2, '-wd':-2*wd/2, '-wd/2':-1*wd/2, '-K': -self.sys.K,
                    'K': self.sys.K, 'wd/2':1*wd/2, 'wd':2*wd/2, '3wd/2': 3*wd/2, '2wd': 4*wd/2, '5wd/2': 5*wd/2}
    
    
    
    def get_kappas(self, bare_kappa, scale_kappa):
        # TODO: modify kappa for drive frequency
        freqs = self.freqs
        freqs_keys = self.freqs.keys()
        freqs_keys_l = list(freqs_keys)
        kappas = {}
    
        for i in range(len(freqs_keys_l)):
            if freqs_keys_l[i] in scale_kappa.keys():
                kappas[freqs_keys_l[i]] = bare_kappa*scale_kappa[freqs_keys_l[i]]
            else:
                kappas[freqs_keys_l[i]] = bare_kappa*1
        return kappas
    
    def get_temps(self, bare_temp, scale_temp):
        
        # TODO: modify temp for drive frequency
        freqs = self.freqs
        freqs_keys = self.freqs.keys()
        freqs_keys_l = list(freqs_keys)
        temps = {}
    
        for i in range(len(freqs_keys_l)):
            if freqs_keys_l[i] in scale_temp.keys():
                temps[freqs_keys_l[i]] = bare_temp*scale_temp[freqs_keys_l[i]]
            else:
                temps[freqs_keys_l[i]] = bare_temp*1
        return temps
        
    
    def get_nbars(self, fixed = False):
        freqs = self.freqs
        freqs_keys = self.freqs.keys()
        freqs_keys_l = list(self.freqs.keys())
        temps = self.temps
        nbars = {}
        
        for i in range(len(freqs_keys_l)):
            
            if fixed == False:
            
                nbars[freqs_keys_l[i]] = nb(np.abs(freqs[freqs_keys_l[i]]*1e6), temps[freqs_keys_l[i]])
            elif fixed == True:  
                if freqs_keys_l[i] == 'K' or freqs_keys_l[i] == '-K':
                    nbars[freqs_keys_l[i]] = 325/35.
                else:
                    nbars[freqs_keys_l[i]] = nb(np.abs(freqs[freqs_keys_l[i]]*1e6), temps[freqs_keys_l[i]])
                    
#         print (nbars)
        return nbars

    @property
    def n_fock(self):
        return self.sys.n_fock
    @property
    def wd(self):
        return self.sys.wd
    @property
    def g3(self):
        return self.sys.g3
    @property
    def g4(self):
        return self.sys.g4
    @property
    def e2(self):
        return self.sys.e2
    @property
    def Pi(self):
        return self.sys.Pi
    @property
    def K(self):
        return self.sys.K
    
    
    def get_c_ops_order(self, max_order):
        freqs = self.freqs
        freqs_keys_l = list(freqs.keys())
        c_ops_order = []#[[qt.Qobj(np.zeros((self.n_fock, self.n_fock)))]*(max_order+1)]*len(freqs)
        
        for i in range(len(freqs)):
            
            c_ops_orderi = []
            
            for j in range(max_order+1):
                
#                 if freqs_keys_l[i] == '5wd/2' and j == 2:
#                     print (self.get_c_op(j, freqs[freqs_keys_l[i]]), 'doing this')
                
                c_ops_orderi.append(self.get_c_op(j, freqs[freqs_keys_l[i]]))
        
            c_ops_order.append(c_ops_orderi)
                
        return c_ops_order
    
    def get_c_ops_all(self, max_order):
        freqs = self.freqs
        freqs_keys = freqs.keys()
        freqs_keys_l = list(freqs.keys())
        c_ops_all = []
        kappas = self.kappas
        nbars = self.nbars
        K = self.K
        for i in range(len(freqs)):
            
            if freqs[freqs_keys_l[i]] != None:
                # TO CHECK
                c_ops_all_op_i = sum(self.get_c_ops_order(max_order)[i][:max_order+1])
                
                if freqs[freqs_keys_l[i]] < 0:
#                     print (freqs_keys_l[i], kappas[freqs_keys_l[i]]/K, nbars[freqs_keys_l[i]], c_ops_all_op[i])
                    c_ops_all.append(np.sqrt(kappas[freqs_keys_l[i]]/K * (1. + nbars[freqs_keys_l[i]]))* c_ops_all_op_i)
    
                elif freqs[freqs_keys_l[i]] > 0:
                    c_ops_all.append(np.sqrt(kappas[freqs_keys_l[i]]/K * (nbars[freqs_keys_l[i]]))* c_ops_all_op_i)
            
#             if freqs_keys_l[i] == '5wd/2':
#                 print (c_ops_all[i])
        return c_ops_all    
       
    
    def get_c_op(self, order, freq):
        wd = self.wd
        e2 = self.e2
        g3 = self.g3
        g4 = self.g4
        Pi = self.Pi
        K = self.K
        a = self.sys.a
#         a = qt.destroy(self.n_fock)
        adag = a.dag()
        
        if order == 0:
            if freq == -self.wd/2:
                return a
            
            elif freq == self.wd/2:
                return adag
            
            else:
                return qt.Qobj(np.zeros((self.sys.N_max, self.sys.N_max)))
        elif order == 1:
            # inductive
#             if freq == -K:
#                 return 8*g3/wd * adag * a
            
#             elif freq == K:
#                 return 8*g3/wd * adag * a
            
            
            if freq == -wd/2:
                return 2*e2/wd * adag
            
            elif freq == wd/2:
                return 2*e2/wd * a
        
            elif freq == -3*wd/2:
                return 3*e2/wd * a
            
            elif freq == 3*wd/2:
                return 3*e2/wd * adag
            
            if freq == -wd:
                return 8*g3/3/wd * a**2
            
            elif freq == wd:
                return 8*g3/3/wd * adag**2
            
            else:
                return qt.Qobj(np.zeros((self.sys.N_max, self.sys.N_max)))
                
        
        elif order == 2:
            if freq == -K:
                return 32*g3**2/(wd**2) * Pi * a**2
                
            elif freq == K:
                return 32*g3**2/(wd**2) * Pi * adag**2
                
            elif freq == -wd/2:
                return -(35/2*g3**2/wd**2 - 6*g4/wd)* a * Pi**2 - (152/9* g3**2/wd**2 - 3*g4/wd)*adag*a**2 - (152/9* g3**2/wd**2 - 3*g4/wd)*a 
            
            elif freq == wd/2:
                return -(35/2*g3**2/wd**2 - 6*g4/wd)* adag * Pi**2 - (152/9 *g3**2/wd**2 - 3*g4/wd)*adag**2*a - (152/9 *g3**2/wd**2 - 3*g4/wd)*adag 
            
            elif freq == -wd:
                return -(592/9 * g3**2 / wd**2 - 16*g4/wd)*Pi*adag*a
            
            elif freq == wd:
                return -(592/9 * g3**2 / wd**2 - 16*g4/wd)*Pi*adag*a
            
            elif freq == -3*wd/2:
                return -(51/5*g3**2/wd**2 - 9/2 * g4/wd) * adag*Pi**2 -(-4*g3**2/wd**2 - 3*g4/wd)*a**3
            
            elif freq == 3*wd/2:
                return -(51/5*g3**2/wd**2 - 9/2 * g4/wd) * a*Pi**2 -(-4*g3**2/wd**2 - 3*g4/wd)*adag**3
            
            elif freq == -4*wd/2:
                return (225/45 * g3**2/(wd**2) + 16/5 * g4/wd) * adag**2 * Pi
            
            elif freq == 4*wd/2:
                return (225/45 * g3**2/(wd**2) + 16/5 * g4/wd) * a**2 * Pi
            
            elif freq == -5*wd/2:
                return (-19/9 * g3**2/wd**2 + 5*g4/2/wd)*a*Pi**2
            
            elif freq == 5*wd/2:
                return (-19/9 * g3**2/wd**2 + 5*g4/2/wd)*adag*Pi**2
            else:
                return qt.Qobj(np.zeros((self.sys.N_max, self.sys.N_max)))


# def get_sorted_eigenstates(n_fock=100, kerr=1, alpha=2, detuning=0):
#     '''
#     Sort the eigenstates of the Kerr-cat Hamiltonian according to
#     decreasing energy. Within degenerate subspaces, sort the eigenstates
#     according to decreasing parity.
#     '''
#     if n_fock % 2 == 1:
#         raise Exception('n_fock cannot be odd.')
#     minus_H0 = -get_hamiltonian(n_fock, kerr, alpha, detuning) # minus sign adopted for proper sorting
#     a = qt.destroy(n_fock)
#     minus_parity_op = -(1j * np.pi * qt.num(n_fock)).expm() # minus sign adopted for proper sorting
#     eigenvalues, eigenstates = qt.simdiag([minus_parity_op, minus_H0], evals=True) # Prioritize the parity operator in diagonalization
    
#     # reorder the eigenstates and eigenvalues according to decreasing energy, and then decreasing parity.
#     parities = -eigenvalues[0].reshape(2, n_fock//2).T.reshape(1, -1)
#     energies = -eigenvalues[1].reshape(2, n_fock//2).T.reshape(1, -1)
    
#     # the phase of the eigenstates are not determined
#     plus_eigenstates = eigenstates[:n_fock//2]
#     minus_eigenstates = eigenstates[n_fock//2:]
#     reordered_eigenstates = [state for pair in zip(plus_eigenstates, minus_eigenstates) for state in pair]
#     return parities, energies, reordered_eigenstates


# def get_transformation_map(eigenstates):
#     '''
#     Returns a matrix which brings a state in the (possibly smaller) eigenbasis
#     representation to the Fock basis representation.
    
#     For states:
#     fock_rep = tf_map * eigenbasis_rep
#     eigenbasis_rep = tf_map.dag() * fock_rep
    
#     For operators:
#     eigenbasis_rep = tf_map.dag() * fock_rep * tf_map
#     fock_rep = tf_map * fock_rep * tf_map.dag()
#     '''
#     tf_map = qt.Qobj(np.array([eigenstate.full()[:, 0] for eigenstate in eigenstates]).T)
#     return tf_map


# def get_transformed_operator(operator, tf_map):
#     return tf_map.dag() * operator * tf_map


# def get_untransformed_operator(operator, tf_map):
#     return tf_map * operator * tf_map.dag()


# ########## Non-essential definitions ##############
# def get_cat_state(n_fock, alpha, parity):
#     return (qt.coherent(n_fock, alpha=alpha) + (-1)**parity * qt.coherent(n_fock, alpha=-alpha)).unit()


# def get_e_ops(n_fock, alpha):
#     cat_p = get_cat_state(n_fock, alpha, 0)
#     cat_m = get_cat_state(n_fock, alpha, 1)
#     coherent_p = (cat_p + cat_m).unit()
#     coherent_m = (cat_p - cat_m).unit()
#     cat_i_p = (coherent_p + 1j * coherent_m).unit()
#     cat_i_m = (coherent_p - 1j * coherent_m).unit()
#     e_ops = [cat_p * cat_p.dag(), cat_m * cat_m.dag(),
#              coherent_p * coherent_p.dag(), coherent_m * coherent_m.dag(),
#              cat_i_p * cat_i_p.dag(), cat_i_m * cat_i_m.dag()]
#     return e_ops


# def get_initial_states(n_fock, alpha):
#     return get_e_ops(n_fock, alpha)

# def expmsolve(lindbladian, rho0, tmax, tdivs, e_ops):
#     delta_t = tmax / tdivs
#     times = np.linspace(0, tmax, tdivs + 1)
    
#     superoperator = (lindbladian * delta_t).expm()
#     qt_result = qt.solver.Result()
    
#     qt_result.states = [rho0]
#     qt_result.times = times
#     qt_result.solver = 'expm'
    
#     rho = rho0
#     for i in range(tdivs):
#         rho = superoperator(rho)
#         qt_result.states.append(rho)
    
    
#     qt_result.expect = qt.expect(e_ops, qt_result.states)
#     qt_result.num_expect = len(qt_result.expect)
#     qt_result.num_collapse = 0
    
#     return qt_result


# def get_prob_state(result, state):
#     return np.array([np.real(rho.overlap(state)) for rho in result.states])


# def fit_exponential(times, y, t_begin=None, t_end=None):
#     if t_begin is None:
#         t_begin = 0
#     if t_end is None:
#         t_end = times[-1]
#     select = np.logical_and(times >= t_begin, times <= t_end)
#     times_select = times[select]
#     y_select = y[select]
    
#     f = lambda t, r, A, C: A * np.exp(- r * t) + C
#     r_guess = 1 / (times[-1] - times[0])
#     p0 = (r_guess, 0.5, 0.5)
    
#     fit_result = curve_fit(f, times_select, y_select, p0=p0)
#     return fit_result[0], f



