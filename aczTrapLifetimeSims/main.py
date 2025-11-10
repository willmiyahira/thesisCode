import numpy as np
import functions as func
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
# import scienceplots
# plt.style.use(['science','nature','grid'])

'''

This script imports a 2d array of the potential and does the following:
    1) interpolation of the potential to a finer grid
    2) takes the gradient of the potential to give a force
    3) calculates the trajectory of the particle using the RK4 algorithm
        i) initial velocity sampled from the Maxwell-Boltzmann distribution
       ii) initial position randomly selected within potential
    4) if the particle ever leaves the potential bounds we mark its lifetime
    5) repeat for N particles 

'''



m = 1.44316060e-25       # mass of rubidium 87 (kg)
kB = 1.38064852e-23     # Boltzmann's constant (J/K)
kB = kB/1e6     # J/uK

## load in the potential (from matlab GUI)
U   = np.loadtxt("aczTempRaw_19p5MHz_phase121.txt", delimiter=',')*kB
x   = np.loadtxt("xraw.txt", delimiter=',')
y   = np.loadtxt("yraw.txt", delimiter=',') - 50e-6

print(y[0]*1e6 + 50)

## turn gravity on (1) or off (0)
gravityON = 1
if gravityON == 1:
    for i in range(len(y)):
        U[i,:] -= m*9.81*y[i]
        
    ind_coarse = [97, 236]
    
else:
    # get indecies at potential minimum
    ind_coarse = np.unravel_index(np.argmin(U, axis=None), U.shape)
    print( x[ind_coarse[1]]*1e6)
    print( y[ind_coarse[0]]*1e6)
        
        
plt.subplots(dpi=300)
plt.grid()
plt.contourf(x*1e6, y*1e6, U/kB, 50, cmap='jet')
plt.clim(0,500)
plt.tight_layout()
plt.show()

plt.subplots(dpi=300)
plt.grid()
xcutidx = 236
plt.plot(y*1e6, U[:,xcutidx]/kB)
plt.ylim(0,40)
plt.tight_layout()
plt.show()
print(np.min(U[0:130,xcutidx]/kB))
print(y[np.argmin(U[0:130,xcutidx]/kB)]*1e6)

# U = np.loadtxt("../../practiceSHOpotential.txt", delimiter=',')
# x, y = np.loadtxt("../../practice_xy.txt", delimiter=',')



# offset positions so the minimum is at (x,y) = (0,0)
y = y - y[ind_coarse[0]]
x = x - x[ind_coarse[1]]

# offset the potential so that the minimum sits at zero
U = U - U[ind_coarse[0], ind_coarse[1]]

# plot potential 
plt.subplots(dpi=300)
plt.subplot(2,1,1)
plt.grid()
plt.plot(x*1e6, U[ind_coarse[0],:]/kB, linewidth=1.5)
plt.xlabel("x ($\mu$m)")
plt.ylabel("ACZ energy ($\mu$K)")
plt.xlim(x[0]*1e6, x[-1]*1e6)
plt.ylim(0,60)
plt.axvline(125, color='k', linestyle='--', linewidth=1)
plt.axvline(-125, color='k', linestyle='--', linewidth=1)

plt.subplot(2,1,2)
plt.grid()
plt.plot(y*1e6, U[:,ind_coarse[1]]/kB, linewidth=1.5)
plt.ylim(0,30)
plt.xlabel("y ($\mu$m)")
plt.ylabel("ACZ energy ($\mu$K)")
plt.xlim(y[0]*1e6, y[-1]*1e6)
plt.axvline(125, color='k', linestyle='--', linewidth=1, label='cutoff')
plt.legend(framealpha=0)

plt.tight_layout()
# plt.savefig("xyPotentials_19p5MHz.pdf")
plt.show()
##########################################################
#%%
## Step 1: interpolate the potential to a finer grid size
numpoints = 20000
U2, x2, y2 = func.interpolatePotential(x,y,U,numpoints)

if gravityON ==0:
    ind = np.unravel_index(np.argmin(U2, axis=None), U2.shape)
else:
    ind = np.unravel_index(np.argmin(U2[0:8000,:], axis=None), U2[0:8000,:].shape)
    
    
y2 = y2 - y2[ind[0]]
x2 = x2 - x2[ind[1]]
U2 = U2 - U2[ind[0], ind[1]]

##### Calculate the trap frequency
xc = 20
yc = 20
mid = numpoints//2
xcut = x2[ind[1]-xc:ind[1]+xc]
ycut = y2[ind[0]-yc:ind[0]+yc]
U2xc = U2[ind[0], ind[1]-xc:ind[1]+xc]
U2yc = U2[ind[0]-yc:ind[0]+yc, ind[1]]
def harmonic(x,a,b,c):
    return a*x**2 + b*x + c
xparams, xerr = curve_fit(harmonic, xcut, U2xc, p0=[1e10,10,0])
yparams, yerr = curve_fit(harmonic, ycut, U2yc, p0=[1e10,10,0])

plt.subplots(dpi=300)
plt.subplot(2,1,1)
xfit = np.linspace(xcut[0], xcut[-1], 1000)
plt.plot(xcut*1e6, U2xc, '.')
plt.plot(xfit*1e6, harmonic(xfit, *xparams), 'r--')
plt.title("x fit")
plt.subplot(2,1,2)
xfit = np.linspace(ycut[0], ycut[-1], 1000)
plt.plot(ycut*1e6, U2yc, '.')
plt.plot(xfit*1e6, harmonic(xfit, *yparams), 'r--')
plt.title("y fit")
plt.tight_layout()
plt.show()

# calculate trap freqs
xfreq = np.sqrt(2*xparams[0]/m)/2/np.pi
yfreq = np.sqrt(2*yparams[0]/m)/2/np.pi
print(xfreq)
print(yfreq)
omega0 = 2*np.pi*(xfreq+yfreq)/2

#%%
## Step 2: take the gradient of the interpolated potential to get a force
Fx, Fy = func.gradientPotential2(U,x,y)
Fxslice = Fx[ind_coarse[0], :]
Fyslice = Fy[:,ind_coarse[1]]

plt.subplots(dpi=300, figsize=(6,4))
plt.subplot(2,1,1)
plt.plot(x*1e6, Fx[ind_coarse[0], :], '-')
plt.title("Fx")

plt.subplot(2,1,2)
plt.plot(y*1e6, Fy[:,ind_coarse[1]], '-')
plt.title("Fy")
# plt.ylim(-0.1e-22,0.1e-22)

plt.tight_layout()
plt.show()


#%% generate initial position and velocity
def generateMBvelocity(mass, temp):
# =============================================================================
# Returns a velocity sampled from the Maxwell-Boltzmann distribution
# Inputs:
#   1) mass = mass of the atom   
#   2) temp = temperature of the atoms
# =============================================================================
    kB = 1.38064852e-23     # Boltzmann's constant (J/K)
    mu = 0
    sigma = np.sqrt(kB*temp/mass)
    return np.random.normal(mu, sigma,1)


temperature = 2e-6     # in Kelvin
startsig = np.sqrt(kB*temperature*1e6/m/omega0**2)

x0 = np.random.normal(0, startsig,1)
vx0 = func.generateMBvelocity(m, temperature)


N = 10000
x0arr = []
vx0arr = []
y0arr = []
vy0arr = []
varr = []
for n in range(N):
    x0arr.append(np.random.normal(0, startsig,1))
    vx0arr.append(generateMBvelocity(m, temperature))
    y0arr.append(np.random.normal(0, startsig,1))
    vy0arr.append(generateMBvelocity(m, temperature))
    varr.append(np.sqrt(vx0arr[n]**2 + vy0arr[n]**2))
    
    
# histograms
## look at the distribution of initial starting positions
def gaussian(x,a,b,c,d):
    return a*np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d
    
# plot histogram of initial X and Y positions
print(startsig)

# Histogram settings
from scipy.stats import norm
bins = 25
range_um = 4 * startsig * 1e6
bin_vals_x, bin_edges_x = np.histogram(x0arr, bins=bins, range=(-range_um, range_um))
bin_vals_y, bin_edges_y = np.histogram(y0arr, bins=bins, range=(-range_um, range_um))
bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])

# Gaussian PDF scaled by total number of samples and bin width
x_fine = np.linspace(-range_um, range_um, 1000)
pdf_x = norm.pdf(x_fine, loc=0, scale=startsig * 1e6)
pdf_x *= len(x0arr) * (bin_edges_x[1] - bin_edges_x[0])

pdf_y = norm.pdf(x_fine, loc=0, scale=startsig * 1e6)
pdf_y *= len(y0arr) * (bin_edges_y[1] - bin_edges_y[0])

fig, axs = plt.subplots(1, 2, layout='constrained', dpi=300, figsize=(4,3))

# plt.hist(lifetimes, 25, density=False, color='skyblue' , edgecolor='black', linewidth=0.5)
axs[0].hist(np.array(x0arr)*1e6, 25, density=False, color='skyblue' , edgecolor='black', linewidth=0.25)
axs[0].plot(x_fine, pdf_x, 'k-', label='Expected Gaussian')
axs[0].axvline(-startsig*1e6, color='k', linestyle='--')
axs[0].axvline(startsig*1e6, color='k', linestyle='--')
axs[0].grid(False)
axs[0].set_xlim(-startsig*1e6*4, startsig*1e6*4)
axs[0].set_xlabel("Starting x ($\mu$m)")
axs[0].set_ylabel("Number of Atoms")

axs[1].hist(np.array(y0arr)*1e6, 25, density=False, color='skyblue' , edgecolor='black', linewidth=0.25)
axs[1].plot(x_fine, pdf_y, 'k-', label='Expected Gaussian')
axs[1].axvline(-startsig*1e6, color='k', linestyle='--')
axs[1].axvline(startsig*1e6, color='k', linestyle='--')
axs[1].grid(False)
axs[1].set_xlim(-startsig*1e6*4, startsig*1e6*4)
axs[1].set_xlabel("Starting y ($\mu$m)")

fig.suptitle("Initial Starting Position (" + str(N) + " Atoms)")

plt.show()
    
#%% look at the distribution of initial starting velocities    
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


varr = np.array(varr)

vend = np.max(vx0arr) + np.mean(vx0arr)
vend = 0.060
v_theory = np.linspace(0, vend, 10000)

figsz = (4,3)
plt.figure(dpi=300)
plt.hist(varr*1e2, 50, density=True, color='skyblue' , edgecolor='black', linewidth=0.25, label='Sampled Velocities')
plt.plot(v_theory*1e2, maxwellBoltzmann2D(m, temperature, v_theory)/1e2, 'r--', linewidth=2, label='MB Distribution')

plt.xlabel('Velocity (cm/s)')
plt.ylabel('Probability')
plt.title("Sampled velocities (" + str(N) + " atoms)")

plt.xlim(0,6)
# plt.ylim(0,0.065)

plt.legend()
plt.minorticks_on()
plt.grid()

plt.tight_layout()
plt.show()    
    
    
#%% try interpolating force along line
x2slice = 0
Fxslice1 = 0

cut1 = 0
cut2 = 1000
Fxslice1 = Fxslice[cut1:cut2]
x2slice = x[cut1:cut2]

xnew = np.linspace(x2slice[0], x2slice[-1], 100000)
# ynew = np.interp(xnew, x2slice, Fxslice1)

xnew2 = 10e-6
ynew2 = np.interp(xnew2, x, Fxslice)

plt.subplots(dpi=300)

plt.plot(x2slice*1e6, Fxslice1, '.')
# plt.plot(xnew*1e6, ynew, 'k-')
plt.plot(xnew2*1e6, ynew2, 'ro')


    
plt.tight_layout()
plt.show()
    
#%% interpolate 2d array
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

x = x2
y = y2
U = U2

# get the spacing between points. Assume an equally spaced grid
dy = y[1]-y[0]
dx = x[1]-x[0]

# take the gradient and add a minus sign to get the force
ygrad, xgrad = np.gradient(U, dy, dx)
Fx = -xgrad
Fy = -ygrad
    

interp_splineX = RectBivariateSpline(y, x, Fx) # y,x
interp_splineY = RectBivariateSpline(y, x, Fy) # y,x

Uinterp = RectBivariateSpline(y, x, U) # y,x
xnew = 5e-6
ynew = -10e-6
    



#%% RK4 implementation
import time
from numba import njit


# =============================================================================
# Using the RK4 method to solve the equations of motion for a general force F
def f(t,y,v):
    return v
# for solving v'(t) = y''(t) = -omega^2 y(t) = f2(t,y,v)
def g(t,y,v,force,mass):
    return force/mass

def RK4(y, v, t, h, force, mass):    
    r1 = f(t,y,v)
    k1 = g(t,y,v,force,mass)
    r2 = f(t+h/2,y+(h*r1)/2,v+(h*k1)/2)
    k2 = g(t+h/2,y+(h*r1)/2,v+(h*k1)/2,force,mass)
    r3 = f(t+h/2,y+(h*r2)/2,v+(h*k2)/2)
    k3 = g(t+h/2,y+(h*r2)/2,v+(h*k2)/2,force,mass)
    r4 = f(t+h,y+(h*r3),v+(h*k3))
    k4 = g(t+h,y+(h*r3),v+(h*k3),force,mass)

    y_next = y + (h/6)*(r1 + 2*r2 + 2*r3 + r4)
    v_next = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    t_next = t + h
    return y_next, v_next, t_next

def generateMBvelocity(mass, temp, outputsize):
# =============================================================================
# Returns a velocity sampled from the Maxwell-Boltzmann distribution
# Inputs:
#   1) mass = mass of the atom   
#   2) temp = temperature of the atoms
# =============================================================================
    kB = 1.38064852e-23     # Boltzmann's constant (J/K)
    mu = 0
    sigma = np.sqrt(kB*temp/mass)
    return np.random.normal(mu, sigma,size=outputsize)

inv_m = 1.0 / m
@njit
def rk4_step(x, vx, y, vy, fx, fy, h, inv_m):
    # RK4 for x
    r1x = vx
    k1x = fx * inv_m
    r2x = vx + 0.5 * h * k1x
    k2x = fx * inv_m
    r3x = vx + 0.5 * h * k2x
    k3x = fx * inv_m
    r4x = vx + h * k3x
    k4x = fx * inv_m

    x_next = x + (h / 6.0) * (r1x + 2*r2x + 2*r3x + r4x)
    vx_next = vx + (h / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)

    # RK4 for y
    r1y = vy
    k1y = fy * inv_m
    r2y = vy + 0.5 * h * k1y
    k2y = fy * inv_m
    r3y = vy + 0.5 * h * k2y
    k3y = fy * inv_m
    r4y = vy + h * k3y
    k4y = fy * inv_m

    y_next = y + (h / 6.0) * (r1y + 2*r2y + 2*r3y + r4y)
    vy_next = vy + (h / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

    return x_next, vx_next, y_next, vy_next

# =============================================================================

# numatoms = 1000
tstart = 0
tend = 50e-3
step = 1e-7
step = 5e-8
numsteps = int((tend - tstart) / step)

temperature = 2.47e-6     # in Kelvin

# ------------------ Simulation Params ------------------
total_atoms = 10000
batch_size = 100
n_batches = total_atoms // batch_size
escape_time_list = []
x0list = []
y0list = []
vx0list = []
vy0list = []
# ------------------------------------------------------
for b in range(n_batches):
    start_time = time.time()
    print(f"Simulating batch {b+1}/{n_batches}")

    # step 0) initialize arrays
    xarr = np.zeros((numsteps, batch_size))
    yarr = np.zeros((numsteps, batch_size))
    vxarr = np.zeros((numsteps, batch_size))
    vyarr = np.zeros((numsteps, batch_size))
    tarr = np.zeros((numsteps, batch_size))
    Earr = np.zeros((numsteps, batch_size))
    
    # step 1) get initial x/y position and velocity
    omega0 = 2*np.pi*(xfreq+yfreq)/2
    print("freq = " + str(round(omega0/2/np.pi,2)))
    print("temp = " + str(temperature*1e6))
    startsigX = np.sqrt(kB*temperature*1e6/m/((2*np.pi*xfreq)**2))
    startsigY = np.sqrt(kB*temperature*1e6/m/((2*np.pi*yfreq)**2))
    x0 = np.random.normal(0, startsigX, size=batch_size)
    y0 = np.random.normal(0, startsigY, size=batch_size)
    
    vx0 = generateMBvelocity(m, temperature, batch_size)
    vy0 = generateMBvelocity(m, temperature, batch_size)
    
    x0list += list(x0)
    y0list += list(y0)
    vx0list += list(vx0)
    vy0list += list(vy0)
    
    # save initial positions and velocities
    np.savez("initial_conditions_19p5MHz_phase121Gravity.npz", x0=np.array(x0list), y0=np.array(y0list), vx0=np.array(vx0list), vy0=np.array(vy0list))
    
    # x0 = 50e-6
    # y0 = 70e-6
    # vx0 = 1e-3
    # vy0 = 10e-3
    
    xarr[0] = x0
    yarr[0] = y0
    vxarr[0] = vx0
    vyarr[0] = vy0
    tarr[0] = tstart
    
    x_max = 125e-6
    y_max = 125e-6
    r_max = np.sqrt(x_max**2 + y_max**2)
    escape_time = np.full(batch_size, np.nan)
    
    
    def energy(x,y,vx,vy):
        return float(Uinterp(y, x)) + 0.5*m*(vx**2 + vy**2)
    
    from scipy.interpolate import RegularGridInterpolator
    
    Fx_interp = RegularGridInterpolator((y, x), Fx, bounds_error=False, fill_value=0.0)
    Fy_interp = RegularGridInterpolator((y, x), Fy, bounds_error=False, fill_value=0.0)
    
    for s in range(numsteps-1):
        
        xold = xarr[s]
        vxold = vxarr[s]
        yold = yarr[s]
        vyold = vyarr[s]
        told = tarr[s]
        
        # evaluate force at current position
        pos = np.column_stack((yold, xold))  # shape (N,2)
        forceX = Fx_interp(pos)
        forceY = Fy_interp(pos)
        
        xarr[s+1], vxarr[s+1], yarr[s+1], vyarr[s+1] = rk4_step(xold, vxold, yold, vyold, forceX, forceY, step, inv_m)
        
        # check if atom has left the trapping region
        escaped = xarr[s+1]**2 + yarr[s+1]**2 > r_max**2
        if escaped.any():
            # print(escaped)
            escape_time = tarr[s][escaped]
            escape_time_list += list(escape_time)
            # print(escape_time)
        
            # For atoms that escaped stop them from being included further down the line
            xarr[s+1][escaped] = np.nan
            yarr[s+1][escaped] = np.nan
            vxarr[s+1][escaped] = 0.0
            vyarr[s+1][escaped] = 0.0
                    
        tarr[s+1] = told + step
                
    end_time = time.time()
    print(f"Batch time: {end_time - start_time:.6f} seconds")

# save data
np.savez("escape_time_list_19p5MHz_phase121Gravity.npz", times=np.array(escape_time_list))
np.savez("t_19p5MHz_phase121Gravity.npz", t=tarr[:,0])

# plot trajectories
# omega0 = 2*np.pi*500
t_theory2 = np.linspace(tarr[0], tarr[-1], 1000)
y_theory2 = yarr[0]*np.cos(omega0*t_theory2) + (vyarr[0]/omega0)*np.sin(omega0*t_theory2)
x_theory2 = xarr[0]*np.cos(omega0*t_theory2) + (vxarr[0]/omega0)*np.sin(omega0*t_theory2)

tarr = np.array(tarr)
xarr = np.array(xarr)
yarr = np.array(yarr)
vxarr = np.array(vxarr)
vyarr = np.array(vyarr)
Earr = np.array(Earr)


plt.subplots(dpi=300)

plt.subplot(2,1,1)
plt.plot(tarr*1e3, xarr*1e6)
plt.plot(t_theory2*1e3, x_theory2*1e6, 'r--')
plt.ylabel("x ($\mu$m)")
plt.grid()
plt.xticks(np.arange(tstart*1e3, tend*1e3, 5), labels=[])
plt.xlim(tstart*1e3, tend*1e3)

plt.subplot(2,1,2)
plt.plot(tarr*1e3, yarr*1e6)
plt.plot(t_theory2*1e3, y_theory2*1e6, 'r--')
plt.ylabel("y ($\mu$m)")
plt.xlabel("time (ms)")
plt.grid()
plt.xticks(np.arange(tstart*1e3, tend*1e3, 5))
plt.xlim(tstart*1e3, tend*1e3)

plt.tight_layout()
plt.show()

plt.subplots(dpi=300)

plt.subplot(2,1,1)
plt.plot(tarr*1e3, Earr/kB)
plt.ylabel("energy ($\mu$K)")
plt.grid()


plt.subplot(2,1,2)
plt.plot(tarr*1e3, (Earr/Earr[0] - 1)*100)

plt.xlabel("time (ms)")
plt.ylabel("energy change ($\%$)")
plt.grid()


plt.tight_layout()
plt.show()








