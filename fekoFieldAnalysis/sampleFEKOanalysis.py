import numpy as np
import pandas as pd


## specify the grid size in each spatial dimension (given in PostFEKO)
num_x = 101
num_y = 1
num_z = 101

## directory where files are kept and file names
root = "filepath"
filename = "bfield.hfe"


def data_import(filename, num_x, num_y, num_z, fieldtype):
    start_row = 16
    end_row = start_row + num_x * num_y * num_z - 1
    
    # Define column widths and read the file
    colspecs = [(0, 33), (33, 51), (51, 69), (69, 87), (87, 105), (105, 123), (123, 141), (141, 159), (159, 177)]
    data = pd.read_fwf(filename, colspecs=colspecs, skiprows=start_row - 1, nrows=end_row - start_row + 1)
    
    # Extract x, y, z, and magnetic field components
    x = np.unique(data.iloc[:, 0].values)
    y = np.unique(data.iloc[:, 1].values)
    z = np.unique(data.iloc[:, 2].values)
    
    # Magnetic field components
    hx_vec = data.iloc[:, 3].values + 1j * data.iloc[:, 4].values
    hy_vec = data.iloc[:, 5].values + 1j * data.iloc[:, 6].values
    hz_vec = data.iloc[:, 7].values + 1j * data.iloc[:, 8].values
    
    # Reshape the field data
    hx = np.reshape(hx_vec, (len(z), len(y)))
    hy = np.reshape(hy_vec, (len(z), len(y)))
    hz = np.reshape(hz_vec, (len(z), len(y)))
    
    # Calculate B field components
    u = (4 * np.pi) * 1e-7  # Vacuum permeability (T*m/A)
    if fieldtype == 'magnetic':
        bx = u * hx
        by = u * hy
        bz = u * hz
        return x, y, z, bx, by, bz
    
    else:
        bx = hx
        by = hy
        bz = hz
        return x, y, z, bx, by, bz   

def get_e_and_temp(bm, detuning):
    """Calculate energy, temperature"""
    # Constants
    fp = 2
    m = 2
    # fm = 1
    mprime = 1
    i = 3 / 2

    # g = 9.81
    hbar = 1.054571596e-34  # Planck's constant (J*s)
    ub = 9.274009994e-24    # Bohr magneton (J/T)
    k = 1.38064852e-23      # Boltzmann constant (J/K)
    # m_rb = 1.44316060e-25   # Mass of rubidium-87 (kg)

    sp = hbar * np.sqrt((fp + m) * (fp + mprime)) / (2 * i + 1)

    # Calculate Rabi frequency
    omega = -(ub / hbar**2) * (sp * bm)

    # Detuning in radians/sec
    detuning = detuning * 2 * np.pi * 1e6

    # Energy calculation
    e = (hbar / 2) * (-np.abs(detuning) + np.sqrt(detuning**2 + np.abs(omega)**2))

    # Temperature calculation
    temp = e * (1e6 / k)

    return e, temp


x1, y1, z1, bx, by, bz = data_import(root+filename, num_x, num_y, num_z, 'magnetic')
x = x1*1e6
y = y1*1e2
z = z1*1e6

## convert into ACZ potential
detuning = 1
bminus = bx - 1j * by
eacz, temp =  get_e_and_temp(bminus, detuning)
