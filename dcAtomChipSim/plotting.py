import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

L = np.arange(2,14,2)

data = np.load("atomChipDCsimData_H850um.npy")
collbls = ['Iz (A)', 'Bhold (G)', 'Bioffe (G)', 'L (mm)', 'Iend', 'Imid', 'fx (Hz)', 'fy (Hz)', 'fz (Hz)', 'x0 (um)', 'y0 (um)', 'z0 (um)', 'axial trap depth (uK)']


fig, ax = plt.subplots(dpi=300, figsize=[7,4.5])

for l in range(len(L)):
    
    data2 = pd.DataFrame(data[:,:,l], columns=collbls)
    grouped = data2.groupby("Iend")

    plt.subplot(2,3,l+1)
    for name, group in grouped:
        
        plt.plot(group["Imid"], group["fz (Hz)"],'-', linewidth=1.5, label=str(int(group["Iend"].to_numpy()[0])))
            
    
    plt.title("L = "+str(int(L[l]))+" mm")
    plt.ylim(0,np.max(group["fz (Hz)"])+25)
    plt.xlim(0,12)
    plt.grid(False)

plt.legend(title="I$_{end}$", fontsize=10, title_fontsize=12)

fig.supylabel("Axial Trap Frequency (Hz)", fontsize=12)
fig.supxlabel("I$_{mid}$ (A)", fontsize=12)

plt.tight_layout()
plt.savefig("dcTrapSim_H850um_trapfreq.pdf")
plt.show()


#%%
fig, ax = plt.subplots(dpi=300, figsize=[7,4.5])

for l in range(len(L)):
    
    data2 = pd.DataFrame(data[:,:,l], columns=collbls)
    grouped = data2.groupby("Iend")

    plt.subplot(2,3,l+1)
    for name, group in grouped:
        
        plt.axhline(1, color='k', linestyle='--')
        plt.fill_between(np.linspace(0,12,1000), y1=1, y2=5, color='lightgreen', alpha=0.03, zorder=0)
        
        plt.plot(group["Imid"], group["axial trap depth (uK)"]/1e3,'-', linewidth=1.5, label=str(int(group["Iend"].to_numpy()[0])))
            
    
    plt.title("L = "+str(int(L[l]))+" mm")
    plt.ylim(0,np.max(group["axial trap depth (uK)"]/1e3)+0.25)
    plt.xlim(0,12)
    plt.grid(False)
    
    # plt.xticks()

plt.legend(title="I$_{end}$", fontsize=10, title_fontsize=12)

fig.supylabel("Axial Trap Depth (mK)", fontsize=12)
fig.supxlabel("I$_{mid}$ (A)", fontsize=12)

plt.tight_layout()
plt.savefig("dcTrapSim_H850um_trapdepth.pdf")
plt.show()



