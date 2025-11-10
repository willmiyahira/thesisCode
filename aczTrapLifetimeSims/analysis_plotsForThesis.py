import numpy as np
import functions as func
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import scienceplots
plt.style.use(['science','nature','grid'])

m = 1.44316060e-25       # mass of rubidium 87 (kg)
kB = 1.38064852e-23     # Boltzmann's constant (J/K)
kB = kB/1e6     # J/uK


#%% check initial conditions
# initcond = np.load("initial_conditions_20p5MHz_gravity.npz")
initcond = np.load("initial_conditions_19p5MHz_new.npz")

# positions
x0 = initcond['x0']
y0 = initcond['y0']

def gaussian(x,a,b,c,d):
    return a*np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d
    
# plot histogram of initial X and Y positions
temperature = 2.47e-6     # in Kelvin
omega0 = 2*np.pi*275
startsig = np.sqrt(kB*temperature*1e6/m/omega0**2)
print(startsig)

# Histogram settings
from scipy.stats import norm
bins = 25
range_um = 4 * startsig * 1e6
bin_vals_x, bin_edges_x = np.histogram(x0, bins=bins, range=(-range_um, range_um))
bin_vals_y, bin_edges_y = np.histogram(y0, bins=bins, range=(-range_um, range_um))
bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])

# Gaussian PDF scaled by total number of samples and bin width
x_fine = np.linspace(-range_um, range_um, 1000)
pdf_x = norm.pdf(x_fine, loc=0, scale=startsig * 1e6)
pdf_x *= len(x0) * (bin_edges_x[1] - bin_edges_x[0])

pdf_y = norm.pdf(x_fine, loc=0, scale=startsig * 1e6)
pdf_y *= len(y0) * (bin_edges_y[1] - bin_edges_y[0])

fig, axs = plt.subplots(1, 2, layout='constrained', dpi=300, figsize=(4,3))

# plt.hist(lifetimes, 25, density=False, color='skyblue' , edgecolor='black', linewidth=0.5)
axs[0].hist(np.array(x0)*1e6, 25, density=False, color='skyblue' , edgecolor='black', linewidth=0.25)
axs[0].plot(x_fine, pdf_x, 'k-', label='Expected Gaussian')
axs[0].axvline(-startsig*1e6, color='k', linestyle='--')
axs[0].axvline(startsig*1e6, color='k', linestyle='--')
axs[0].grid(False)
axs[0].set_xlim(-startsig*1e6*4, startsig*1e6*4)
axs[0].set_xlabel("Starting x ($\mu$m)")
axs[0].set_ylabel("Number of Atoms")

axs[1].hist(np.array(y0)*1e6, 25, density=False, color='skyblue' , edgecolor='black', linewidth=0.25)
axs[1].plot(x_fine, pdf_y, 'k-', label='Expected Gaussian')
axs[1].axvline(-startsig*1e6, color='k', linestyle='--')
axs[1].axvline(startsig*1e6, color='k', linestyle='--')
axs[1].grid(False)
axs[1].set_xlim(-startsig*1e6*4, startsig*1e6*4)
axs[1].set_xlabel("Starting y ($\mu$m)")

fig.suptitle("Initial Starting Position (" + str(len(x0)) + " Atoms)")

plt.show()

# velocities
vx0 = initcond['vx0']
vy0 = initcond['vy0']

# Plot the sampled velocities with the 2D MB distribution
def maxwellBoltzmann2D(mass, temp, vel):
    kB = 1.38064852e-23     # J/K
# =============================================================================
# Returns the 2D Maxwell-Boltzmann probability distribution for a given velocity
# Inputs:
#   1) mass = mass of the atom   
#   2) temp = temperature of the atoms
#   3) vel  = array of velocities to calculate the MB distribution at
# =============================================================================
    a = np.sqrt(kB*temp/mass)
    f = (vel/a**2)*np.exp(-0.5*(vel/a)**2)
    return f


varr = np.sqrt(vx0**2 + vy0**2)

vend = 0.060
v_theory = np.linspace(0, vend, 1000)

plt.figure(dpi=300)
plt.hist(varr*1e2, 50, density=True, color='skyblue' , edgecolor='black', linewidth=0.25, label='Sampled Velocities')
plt.plot(v_theory*1e2, maxwellBoltzmann2D(m, temperature, v_theory)/1e2, 'k-', linewidth=2, label='MB Distribution')

plt.xlabel('Velocity (cm/s)')
plt.ylabel('Probability')
plt.title("Sampled velocities (" + str(len(varr)) + " atoms)")

plt.xlim(0,6)
# plt.ylim(0,0.065)

plt.legend()
plt.minorticks_on()
plt.grid()

plt.tight_layout()
plt.show()    

#%% lifetime analysis
lifetimes = np.load("escape_time_list_19p5MHz_new.npz")['times']

numatoms = len(x0)

# tfull = np.load("t_20p5MHz.npz")['t']
tfull = np.load("t_19p5MHz_new.npz")['t']
lifetimes = list(np.sort(lifetimes))
matches = np.isin(tfull, lifetimes)
indices = np.where(matches)[0]
indices = np.append(indices, len(tfull)-1)
atoms = np.ones_like(tfull)*numatoms
for i in range(len(indices)-1):
    atoms[indices[i]:indices[i+1]] -= i+1
    

# fit to decaying exponential
def decay(t,a,b,t0):
    return a*np.exp(-(t-t0)/b)

# fit initial decay with linear fit
def linear(t,m,b):
    return m*t + b

c1 = 45000
c2 = 90000
# c1=0
# c2=-1
tcut = tfull[c1:c2]
atomscut = atoms[c1:c2]

params, err = curve_fit(linear, tcut, atomscut/numatoms)
xfit = np.linspace(tcut[0],tcut[-1],1000)
xfit = np.linspace(0,20e-3,1000)
yfit = linear(xfit, *params)

fig, axs = plt.subplots(dpi=300)
plt.subplot(2,1,1)
# plt.plot(tcut*1e3, atomscut/numatoms, '-')
plt.plot(tfull*1e3, atoms/numatoms, '-', linewidth=1.5, label="Simulation")
plt.plot(xfit*1e3, yfit, 'k--', label="Linear Fit")

plt.xticks(np.arange(0,60,10), labels=[])

# plt.xlabel("Time (ms)")
plt.ylabel("$\%$ of Atoms Left")
plt.xlim(0.00, 50)
# plt.ylim(numatoms-len(lifetimes), numatoms+10)
plt.ylim(0.96,1.005)
plt.grid()

plt.fill_between((tcut[0]*1e3, tcut[-1]*1e3), 0.5,1.5, color='grey', alpha=0.2, label="fitting region")

plt.legend()




## calculate lifetime from linear fit
lifetimeFromFit = (-1/params[0])*1e3
print(f"Lifetime from Fit = {lifetimeFromFit:.2f} ms")

plt.text(12,0.986, r"$\tau = $ " + str(round(lifetimeFromFit,2)) + " ms", fontsize=10)



### with gravity
lifetimes = np.load("escape_time_list_19p5MHz_newGravity.npz")['times']

numatoms = len(x0)

tfull = np.load("t_19p5MHz_newGravity.npz")['t']
lifetimes = list(np.sort(lifetimes))
matches = np.isin(tfull, lifetimes)
indices = np.where(matches)[0]
indices = np.append(indices, len(tfull)-1)
atoms = np.ones_like(tfull)*numatoms
for i in range(len(indices)-1):
    atoms[indices[i]:indices[i+1]] -= i+1
    

# fit to decaying exponential
def decay(t,a,b,t0):
    return a*np.exp(-(t-t0)/b)

# fit initial decay with linear fit
def linear(t,m,b):
    return m*t + b

c1 = 37500
c2 = 65000
# c1=0
# c2=-1
tcut = tfull[c1:c2]
atomscut = atoms[c1:c2]

params, err = curve_fit(linear, tcut, atomscut/numatoms)
xfit = np.linspace(tcut[0],tcut[-1],1000)
xfit = np.linspace(0,20e-3,1000)
yfit = linear(xfit, *params)

plt.subplot(2,1,2)
# plt.plot(tcut*1e3, atomscut/numatoms, '-')
plt.plot(tfull*1e3, atoms/numatoms, '-', linewidth=1.5, label="Simulation \n (with gravity)")
plt.plot(xfit*1e3, yfit, 'k--', label="Linear Fit")

plt.xlabel("Time (ms)")
plt.ylabel("$\%$ of Atoms Left")
plt.xlim(0.00, 50)
# plt.ylim(numatoms-len(lifetimes), numatoms+10)
plt.ylim(0.5,1.05)
plt.grid()

plt.fill_between((tcut[0]*1e3, tcut[-1]*1e3), 0.5,1.5, color='grey', alpha=0.2, label="fitting region")

plt.legend()

lifetimeFromFit = (-1/params[0])*1e3
print(f"Lifetime from Fit = {lifetimeFromFit:.2f} ms")

plt.text(12,0.85, r"$\tau = $ " + str(round(lifetimeFromFit,2)) + " ms", fontsize=10)


# plt.axvline(tcut[0]*1e3)
# plt.axvline(tcut[-1]*1e3)

fig.align_ylabels()
plt.tight_layout()
plt.savefig("aczLifetimeSim_gravityComp.pdf")
plt.show()




#%% phase 121 analysis
lifetimes = np.load("escape_time_list_19p5MHz_phase121Gravity.npz")['times']

numatoms = len(x0)

# tfull = np.load("t_20p5MHz.npz")['t']
tfull = np.load("t_19p5MHz_new.npz")['t']
tfull = np.load("t_19p5MHz_phase121Gravity.npz")['t']


lifetimes = list(np.sort(lifetimes))
matches = np.isin(tfull, lifetimes)
indices = np.where(matches)[0]
indices = np.append(indices, len(tfull)-1)
atoms = np.ones_like(tfull)*numatoms
for i in range(len(indices)-1):
    atoms[indices[i]:indices[i+1]] -= i+1
    

# fit to decaying exponential
def decay(t,a,b,t0):
    return a*np.exp(-(t-t0)/b)

# fit initial decay with linear fit
def linear(t,m,b):
    return m*t + b


c1 = 45000 *2
c2 = 75000 *2
# c1=0
# c2=-1
tcut = tfull[c1:c2]
atomscut = atoms[c1:c2]

params, err = curve_fit(linear, tcut, atomscut/numatoms)
xfit = np.linspace(tcut[0],tcut[-1],1000)
xfit = np.linspace(0,20e-3,1000)
yfit = linear(xfit, *params)

plt.subplots(dpi=300)
# plt.plot(tcut*1e3, atomscut/numatoms, '-')
plt.plot(tfull*1e3, atoms/numatoms, '-', linewidth=1.5, label="Simulation")
plt.plot(xfit*1e3, yfit, 'k--', label="Linear Fit")

plt.xlabel("Time (ms)")
plt.ylabel("$\%$ of Atoms Left")
plt.xlim(0.00, 50)
# plt.ylim(numatoms-len(lifetimes), numatoms+10)
plt.ylim(0.967,1.004)
plt.grid()

plt.text(12,0.99, r"$\tau = $ " + str(round(lifetimeFromFit,2)) + " ms", fontsize=10)



plt.fill_between((tcut[0]*1e3, tcut[-1]*1e3), 0.5,1.5, color='grey', alpha=0.2, label="fitting region")

plt.legend()

# plt.axvline(tcut[0]*1e3)
# plt.axvline(tcut[-1]*1e3)

plt.tight_layout()
plt.savefig("aczLifetimeSim_phase121Gravity.pdf")
plt.show()


## calculate lifetime from linear fit
lifetimeFromFit = (-1/params[0])*1e3
print(f"Lifetime from Fit = {lifetimeFromFit:.2f} ms")






