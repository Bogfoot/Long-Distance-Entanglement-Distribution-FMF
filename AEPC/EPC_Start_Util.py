# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:30:24 2025

@author: Adrian
"""

import subprocess, numpy as np, time
from EPC import PolarizationDevice

def set_voltage(EPC, V):

    """ Apply voltages to all four DAC channels. 
    #NOTE: Decided not to do this -> Might have to change strategy
    depending on results next week. 
    #NOTE: I don't think there's anything to be changed here.
    """

    # TODO: Next aproach would be to selectively probe each crystal
    # individually to see which one has the most "positive" effect towards the
    # target intensity
    np.clip(V, 0, 130)
    EPC.set_voltage("DAC0", V[0])
    EPC.set_voltage("DAC1", V[1])
    EPC.set_voltage("DAC2", V[2])
    EPC.set_voltage("DAC3", V[3])

# TODO: TEST THIS || DONE
print("configure DAC6, DAC3-DAC0")

# Initialize MCP2210CLI with subprocess
subprocess.run(
    ["mcp2210cli", "-spitxfer=28,4f", "-bd=100000,", "-cs=gp4", "-md=1"],
    stdout=subprocess.DEVNULL
)
temperature = 50 #from 10 to 70degC
EPC_1 = PolarizationDevice("0000872235")
EPC_2 = PolarizationDevice("0001005125")

EPC_1.set_temperature(temperature)
EPC_2.set_temperature(temperature)

#%%
V = np.array([120,120,120,120])
V = np.array([65,65,65,65])
# V = np.array([0,0,0,0])


set_voltage(EPC_1, V)
set_voltage(EPC_2, V)


#%% EPC Voltage Sweep Test

import time, subprocess
from EPC import PolarizationDevice

# Initialize MCP2210CLI with subprocess
subprocess.run(
    ["mcp2210cli", "-spitxfer=28,4f", "-bd=100000,", "-cs=gp4", "-md=1"],
    stdout=subprocess.DEVNULL
)
temperature = 50

EPC2 = PolarizationDevice("0001005125")  # controls H2/V2/D2/A2
EPC2.set_temperature(temperature)

# Initialize both EPCs by serial number
EPC1 = PolarizationDevice("0000872235")  # controls H1/V1/D1/A1
EPC1.set_temperature(temperature)

# Voltage sweep parameters
voltages = range(0, 61, 20)
delay_s = 1.             # wait between voltage steps (seconds)

print("Starting EPC voltage sweep test...\n")
#%%
# Sweep all 4 DAC channels on EPC1
for dac in range(4):
    for v in voltages:
        print(f"→ EPC1 DAC{dac}: {v:.1f} V")
        EPC1.set_voltage(f"DAC{dac}", v)
        time.sleep(delay_s)
    print(f"EPC1 DAC{dac} sweep complete.\n")
    # Reset this DAC to 0 V before moving to next channel
    EPC1.set_voltage(f"DAC{dac}", 0.0)
    time.sleep(1.0)

# Sweep all 4 DAC channels on EPC2
for dac in range(4):
    for v in voltages:
        print(f"→ EPC2 DAC{dac}: {v:.1f} V")
        EPC2.set_voltage(f"DAC{dac}", v)
        time.sleep(delay_s)
    print(f"EPC2 DAC{dac} sweep complete.\n")
    EPC2.set_voltage(f"DAC{dac}", 0.0)
    time.sleep(1.0)

print("Voltage sweep test completed for both EPCs.")

