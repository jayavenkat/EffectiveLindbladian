 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:14:35 2023

@author: jaya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:21:55 2023

@author: jaya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:58:40 2023

@author: jaya

"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import qutip as qt
from matplotlib import cm

# Set the figure size in inches
def set_figure_size(width, height, ax=None):
    if not ax:
        ax = plt.gca()

    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    fig_width = float(width) / (right - left)
    fig_height = float(height) / (top - bottom)
    ax.figure.set_size_inches(fig_width, fig_height)

# issue with CMU where minus sign is replaced with hyphen
def replace_minus_with_hyphen(label):
    if isinstance(label, str):
        return label.replace('-', '$-$')
    elif isinstance(label, (list, tuple)):
        return [replace_minus_with_hyphen(l) for l in label]
    return label

# Set the plotting settings
def set_plot_settings(pres_type):
    if pres_type == 'talk':
        s = 16
    elif pres_type == 'paper':
        s = 10
    mpl.rc('font',family='sans-serif')
    mpl.rc('font',size=s)
    mpl.rc('font',size=s)
    mpl.rcParams['font.family'] = 'CMU Sans Serif'
    mpl.rcParams['axes.formatter.useoffset'] = False
    mpl.rcParams['ytick.right'] = False
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['legend.fontsize'] = s
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['axes.labelsize'] = s
    mpl.rcParams['xtick.labelsize'] = s
    mpl.rcParams['ytick.labelsize'] = s
    mpl.rcParams['xtick.major.pad']=  2 #3.5
    mpl.rcParams['ytick.major.pad']=  2 #3.5
    mpl.rcParams['axes.labelpad'] = 1 #4.0
    mpl.rcParams['legend.handlelength'] = 1.0#2.0
    mpl.rcParams['legend.handletextpad'] = 0.4# 0.8
    mpl.rcParams['legend.columnspacing'] = 1.2# 2.0,
    mpl.rcParams['lines.markersize'] = 4.0

# Construct the Hamiltonian of the SHO
def get_hamiltonian(n_fock=100, frequency=0, K = 0, squeezing = 0):
    a = qt.destroy(n_fock)
    a_dag = a.dag()
    H = -frequency * a_dag * a + K * a_dag * a_dag * a * a \
        - squeezing * (a_dag**2) - np.conjugate(squeezing) * a**2
    return H



# # Calculate the energy splittings of the system
# def get_splittings(n_fock=100, frequency=0, n_splitting=3):
#     H = get_hamiltonian(n_fock, frequency)
#     splittings = []
#     h_eigs = H.eigenenergies()[::-1]
#     for i in range(2, 2 * (n_splitting + 1), 2):
#         splittings.append(h_eigs[i + 1] - h_eigs[i])
#     return np.array(splittings)

# Construct a non-Hermitian Hamiltonian of the SHO that un-does the shrinking
def get_nonhermitian_hamiltonian(n_fock=100, frequency=0, \
                                 kappa_loss = 1e-2, kappa_gain = 1e-4):
    a = qt.destroy(n_fock)
    a_dag = a.dag()
    H_nonH = +0*(+1j * kappa_loss/2 * a_dag * a + 1j * kappa_gain/2 * a_dag * a)
    return H_nonH

# Create collapse operators for the system
def get_c_ops(n_fock=100, kappa_loss=1e-2, kappa_gain=1e-4):
    a = qt.destroy(n_fock)
    a_dag = a.dag()
    c_ops = [np.sqrt(kappa_loss) * a, np.sqrt(kappa_gain) * a_dag]
    return c_ops

# Construct the Lindbladian of the system
def get_lindbladian(n_fock=100,frequency=0, kerr = 1, squeezing = 0, \
                    kappa_loss=1e-2, kappa_gain=1e-4):
    a = qt.destroy(n_fock)
    a_dag = a.dag()
    H = get_hamiltonian(n_fock, frequency, kerr, squeezing)
    H_nonH = get_nonhermitian_hamiltonian(n_fock, frequency, kappa_loss, kappa_gain)
    c_ops = get_c_ops(n_fock, kappa_loss, kappa_gain)
    lindbladian = qt.liouvillian(H + H_nonH, c_ops)
    return lindbladian

# Obtain sorted eigenstates of the system according to energy and parity
def get_sorted_eigenstates(n_fock=100, frequency=0, kerr = 1, squeezing = 0):
    if n_fock % 2 == 1:
        raise ValueError('n_fock cannot be odd.')

    minus_H = -get_hamiltonian(n_fock, frequency, kerr, squeezing)
    minus_parity_op = -(1j * np.pi * qt.num(n_fock)).expm()

    eigenvalues, eigenstates = qt.simdiag([minus_parity_op, minus_H], evals=True)

    # reorder the eigenstates and eigenvalues according to decreasing energy, and then decreasing parity.
    parities = -eigenvalues[0].reshape(2, n_fock//2).T.reshape(1, -1)
    energies = -eigenvalues[1].reshape(2, n_fock//2).T.reshape(1, -1)
    
    # the phase of the eigenstates are not determined
    plus_eigenstates = eigenstates[:n_fock//2]
    minus_eigenstates = eigenstates[n_fock//2:]
    reordered_eigenstates = [state for pair in zip(plus_eigenstates, minus_eigenstates) for state in pair]
    return parities, energies, reordered_eigenstates

def find_closest_eigenvalue(eigenvalues):
    closest_eigenvalue = complex(np.inf, np.inf)  # Initialize with an arbitrary complex number
    min_nonzero_real_part = float('inf')

    for eigenvalue in eigenvalues:
        real_part = np.real(eigenvalue)
        imaginary_part = np.imag(eigenvalue)

        if np.isclose(real_part, 0, atol=5e-13) and np.isclose(imaginary_part, 0, atol=5e-13):
            continue

        if (real_part != 0 and
                abs(real_part) < abs(min_nonzero_real_part) and
                abs(imaginary_part) < abs(np.imag(closest_eigenvalue))):
            closest_eigenvalue = eigenvalue
            min_nonzero_real_part = real_part

    return closest_eigenvalue

def smallest_eig_calculator(l_eigs):
    '''
    Parameters
    ----------
    l_eigs : complex
        eigenvalues of the Lindbladian

    Returns
    -------
    real part of the eigenvalue minus of which has the smallest real part ,
    imaginary part of the eigenvalue whose real part has the smallest magnitude

    '''
    # Extract real parts of eigenvalues
    minus_real_parts = -np.real(l_eigs)
    # Extract imaginary parts of eigenvalues
    imag_parts = np.imag(l_eigs)
    
    #print(minus_real_parts)
    
    # Find eigenvalue with smallest nonzero real part
    nonzero_real_parts = minus_real_parts[minus_real_parts > 5e-13 and np.abs(imag_parts) < 1e-13]  # Exclude zero real parts
    smallest_nonzero_real = np.min(nonzero_real_parts)
    
    print(np.min(smallest_nonzero_real))
    
    # Find index of the eigenvalue with smallest nonzero real part
    smallest_nonzero_real_index = np.where(minus_real_parts == smallest_nonzero_real)[0][0]

    # Print the eigenvalue with the smallest nonzero real part
    real_smallest_nonzero_eigenvalue = -minus_real_parts[smallest_nonzero_real_index]
    imag_smallest_nonzero_eigenvalue = imag_parts[smallest_nonzero_real_index]
    
    return real_smallest_nonzero_eigenvalue, imag_smallest_nonzero_eigenvalue