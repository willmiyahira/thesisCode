import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import time


def B(x,y,z,a,b,c,L,theta,I):
    """
    Parameters
    ----------
    x,y,z => location at which to evaluate the magnetic field
    a,b,c => starting x,y,z position of the wire
    L : length of the finite wire
    theta : angle of the wire with respect to the z-axis
    I : current going through the wire

    """
    mu0 = 4*np.pi*1e-3  # vacuum permitivity (G*m/A)
    
    X, Y, Z = np.meshgrid(x,y,z, indexing="ij")
    
    xt = -(Z-c)*np.sin(theta) + (X-a)*np.cos(theta)
    yt = Y - b
    zt = (Z-c)*np.cos(theta) + (X-a)*np.sin(theta)
    
    fact1 = (mu0*I/4/np.pi)*(1/(np.square(xt) + np.square(yt)))
    fact2 = (zt / np.sqrt(np.square(xt) + np.square(yt) + np.square(zt))) + ((L - zt) / np.sqrt(np.square(xt) + np.square(yt) + np.square(L - zt)))
    
    Bx = fact1*fact2*(-yt*np.cos(theta))
    By = fact1*fact2*xt
    Bz = fact1*fact2*(yt*np.sin(theta))
    
    return Bx,By,Bz

def dczEnergy(Bx,By,Bz):
    gauss2tesla = 1e4 # converts gauss to tesla (1T = 1e4 G)(multiply for T->G)
    gF = 0.5
    mF = 2
    # hbar = 1.054571596e-34 # Modified Planck's constant (J*s)
    uB = 9.274009994E-24 # Bohr mageneton (J/T)
    uB = uB/gauss2tesla
    # mRb = 1.44316060e-25 # mass of rubidium 87 (kg)
    kB = (1.38064852e-23)/(1e6)     # Boltzmann's constant (J/uK)
    
    Bmag = np.sqrt(np.square(Bx) + np.square(By) + np.square(Bz));
    
    Udcz = uB*mF*gF*Bmag/kB
    return Udcz

def harmonic(x,a,b,c):
    return a*x**2 + b*x + c
    
def getTrapFreq(xdata, Energy, startparams): 
    params, err = curve_fit(harmonic, xdata, Energy, p0=startparams)
    # params, err = curve_fit(harmonic, xdata, Energy)
    mRb = 1.44316060e-25 # mass of rubidium 87 (kg)
    kB = (1.38064852e-23)/(1e6)     # Boltzmann's constant (J/uK)
    a = params[0]
    trapFreq = np.sqrt(a*2*kB/mRb)/2/np.pi
    
    xfit = np.linspace(xdata[0], xdata[-1],1000)
    # plt.subplots(dpi=300)
    # plt.plot(xdata*1e6, Energy, '.')
    # plt.plot(xfit*1e6, harmonic(xfit, *params), '--')
    # plt.title("Trap Freq = " + str(trapFreq) + " Hz")
    # plt.grid(alpha=0.2, linestyle='-')
    # plt.tight_layout()
    # plt.show()

    return trapFreq
    
############# ATOM CHIP PARAMETERS

# main Z-wire parameters
Lz = 18e-3
LendcapZ = 7.5e-3
Iz = 1

# endcap parameters
# H = 1e-3        # thickness of microwave carrier chip
# L = 10e-3        # separation between lower endcap wires
Lend = 50e-3    # length of lower endcap wires (much larger than chip)

# Iend = 1        # current through outer endcap wires
# Iendmid = -0.1  # current through middle lower endcap wire

# external field parameters
holdmag = 20
ioffemag = 5

############### ATOM CHIP SIM FUNCTION

N = 151
x = np.linspace(-250e-6,250e-6,N)
y = np.linspace(85e-6,115e-6,N)

x = np.linspace(-250e-6,250e-6,N)
y = np.linspace(90e-6,110e-6,N)

def dcAtomChipSim(Iend,Iendmid,L,H):
    z = np.linspace(-L/2 - 0.75e-3, L/2 + 0.75e-3,N)
    X, Y, Z = np.meshgrid(x,y,z)

    ## main Z wire
    Bend1Zx,Bend1Zy,Bend1Zz = B(x,y,z,0,0,-Lz/2,LendcapZ,-np.pi/2,-Iz)
    BmainZx,BmainZy,BmainZz = B(x,y,z,0,0,-Lz/2,Lz,0,Iz)
    Bend2Zx,Bend2Zy,Bend2Zz = B(x,y,z,0,0,Lz/2, LendcapZ,np.pi/2,Iz)
    
    ## endcap wires beneath Z-wire
    Bend1x,Bend1y,Bend1z = B(x,y,z,-Lend/2,-H,-L/2,Lend,np.pi/2,Iend)
    Bend2x,Bend2y,Bend2z = B(x,y,z,-Lend/2,-H,+L/2,Lend,np.pi/2,Iend)
    
    # middle endcap wire
    Bendmidx,Bendmidy,Bendmidz = B(x,y,z,-Lend/2,-H,0,Lend,np.pi/2,-Iendmid)
    
    
    Bhold = holdmag*np.ones_like(BmainZx)
    Bioffe = ioffemag*np.ones_like(BmainZz)
    
    # add fields together
    Bx = Bend1Zx + BmainZx + Bend2Zx + Bend1x + Bend2x + Bendmidx + Bhold
    By = Bend1Zy + BmainZy + Bend1y + Bend2y + Bend2Zy + Bendmidy
    Bz = Bend1Zz + BmainZz + Bend1z + Bend2z+ Bend2Zz + Bendmidz + Bioffe
    
    Udcz = dczEnergy(Bx, By, Bz)
    
    # add in gravity
    g = 9.81
    mRb = 1.44316060e-25 # mass of rubidium 87 (kg)
    kB = 1.38064852e-23 # Boltzmann's constant (J/K)
    kB = kB/(1e6)  # put into J/uK
    gravity = mRb*g*Y/kB
    Udcz = Udcz + gravity
    
    ## enforce that we look at the potential at x=0 and z=0 (i.e. center of the chip)
    ind = [N//2, np.argmin(Udcz[N//2,:,N//2]), N//2]
    
    
    ## if the middle endcap wire is strong enough to create a double well potential, we will just stop the simulation and return nans
    if np.abs(z[np.argmin(Udcz[ind[0], ind[1], :])])*1e3 > 0.1:
        return np.nan*np.zeros(13)
    
    
    # ind = np.unravel_index(np.argmin(Udcz, axis=None), Udcz.shape)
    # ind = list(ind)
    # ind[0] = N//2
    # ind[2] = N//2
    # print(ind)
    
    xmin = x[ind[0]]
    ymin = y[ind[1]]
    zmin = z[ind[2]]
    minpos = np.array([xmin,ymin,zmin])
    # print(minpos*1e6)
    
    #plot potential
    # plt.subplots(dpi=300, figsize=[5,4])
    # plt.subplot(3,1,1)
    # plt.plot(x*1e6, Udcz[:,ind[1],ind[2]])
    # plt.subplot(3,1,2)
    # plt.plot(y*1e6, Udcz[ind[0],:,ind[2]])
    # plt.subplot(3,1,3)
    # plt.plot(z*1e3, Udcz[ind[0],ind[1],:])
    # plt.tight_layout()
    # plt.show()
    
    ## get trap frequencies
    n = 6
    # sometimes the simulation puts the trap not at (x,y,z)=(~0,trapheight,~0), so just make trap freqs nans so it doesn't stop the loop
    if np.abs(x[ind[0]]*1e6)>2 or np.abs(z[ind[2]]*1e3)>0.25:
        trapFreqX = np.nan
        trapFreqY = np.nan
        trapFreqZ = np.nan
        axialtrapdepth = np.nan
    else:
        trapFreqX = getTrapFreq(x[ind[0]-n:ind[0]+n], Udcz[ind[0]-n:ind[0]+n,ind[1],ind[2]], [1e11,-1e5,400])
        trapFreqY = getTrapFreq(y[ind[1]-n:ind[1]+n], Udcz[ind[0],ind[1]-n:ind[1]+n,ind[2]], [1e11,-1e5,400])
        trapFreqZ = getTrapFreq(z[ind[2]-n//2:ind[2]+n//2], Udcz[ind[0], ind[1], ind[2]-n//2:ind[2]+n//2], [1e11,1,400])
        
        ## get axial trap depth
        zslice = Udcz[ind[0],ind[1],:]
        axialtrapdepth = np.max(zslice) - np.min(zslice)
    
    trapFreqs = np.array([trapFreqX,trapFreqY,trapFreqZ])
    
    
    
    ## get magnetic field at trap minimum
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    return np.array([[Iz, holdmag, ioffemag, L*1e3, Iend, Iendmid, trapFreqX,trapFreqY,trapFreqZ, xmin*1e6,ymin*1e6,zmin*1e6, axialtrapdepth]])


###################################
## SIMULATION

# H = np.array([0.75,1,1.25,1.5,2,3,5])*1e-3
# L = np.arange(1,16,1)*1e-3
# Iend = np.arange(1,21,1)
# ratio = np.linspace(0,1,4)


# Iendmid = Iend[0]*ratio[0]
# trapFreqs, minpos, axialtrapdepth = dcAtomChipSim(Iend[0], Iendmid, L[0], H[1])


# array to store all the data
# data = np.zeros((len(ratio), 7, len(Iend), len(L), len(H)))

# for h in tqdm (range(len(H)), desc="Number of H done"):
#     for l in range(len(L)):
#         for i in range(len(Iend)):
#             for j in range(len(ratio)):
                
#                 Iendmid = Iend[i]*ratio[j]
#                 trapFreqs, minpos, axialtrapdepth = dcAtomChipSim(Iend[i], Iendmid, L[l], H[h])
                
#                 data[j, 0:3, i, l, h] = trapFreqs
#                 data[j, 3:6, i, l, h] = minpos
#                 data[j, 6, i, l, h] = axialtrapdepth


#                 ## save data
#                 np.save('atomChipDCsimData.npy', data)

        
# print(trapFreqs)
# print(minpos)
# print(axialtrapdepth)




#%% Fixed height 

H = (0.38 + 2*0.21 + 0.050)*1e-3    # 0.85 mm (atom chip sandwhich)
Iend = np.arange(3,9)

H = 1.5e-3    # 1.5 mm (atom chip sandwhich)
Iend = np.arange(4,12)


L = np.array([2,4,6,8,10,12])*1e-3
L = np.array([10,15,20,25,30])*1e-3 # larger separations

# Iend = [9]
Iendmid = np.linspace(0,12,200)
# ratio = np.linspace(0,2,75)

# L = np.array([2,10])*1e-3
# Iend = np.arange(1,3)
# # Iendmid = np.arange(0,10)
# ratio = np.linspace(0,1,2)


results = np.zeros((len(Iend)*len(Iendmid), 13, len(L)))

t0code = time.time()
for r in tqdm (range(len(L)), desc="Number of L done"):
    count = 0
    for i in range(len(Iend)):
        for j in range(len(Iendmid)):
            # print(r+i+j)
            results[count,:,r] = dcAtomChipSim(Iend[i], Iendmid[j], L[r], H)
            count += 1
            
            ## save data
            np.save('atomChipDCsimData_H1500um_largerL.npy', results)

print("The loop took this many seconds to do: ")
print(time.time()-t0code)

# L = 6e-3
# Iend = 9
# Iendmid = 4.4
            
# results = dcAtomChipSim(Iend, Iendmid, L, H)


#%% copy to clipboard
# import pyperclip
# import pandas as pd

# df = pd.DataFrame(results)

# # Convert the DataFrame to a CSV string (which is clipboard-friendly)
# csv_data = df.to_csv(index=False, header=False, sep='\t')

# # Copy the CSV string to the clipboard
# pyperclip.copy(csv_data)

















