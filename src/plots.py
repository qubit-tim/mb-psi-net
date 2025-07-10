import os
import json

import matplotlib.pyplot as plt

'''
This will be the home of plotting functions for our experiments.
'''

def EnergyGraph(logfile="Jastrow.log", title="", exact_gs_energy=0, color='C8'):
    if not os.path.exists(logfile):
        print(f"The file '{logfile}' does not exist.")
        print("Please run the energy calculation first to generate the log file.\n")
        return
    data = json.load(open(logfile, 'r'))

    iters = data["Energy"]["iters"]
    energy = data["Energy"]["Mean"]["real"]

    fig, ax1 = plt.subplots()
    ax1.plot(iters, energy, color=color, label='Energy (Jastrow)')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    # We might want to set axis limits based on the exact ground state energy...maybe
    #plt.axis([0,iters[-1],exact_gs_energy-0.1,exact_gs_energy+1.0])
    if not exact_gs_energy == 0:
        plt.axhline(y=exact_gs_energy, xmin=0,
                    xmax=iters[-1], linewidth=2, color='k', label='Exact')
    ax1.legend()
    plt.show()
    

