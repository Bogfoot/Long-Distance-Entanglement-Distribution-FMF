import os
from functions import *      # functions file

##USER defined run parameters
PC_channel_0 = 60 #from 0 to 130V
PC_channel_1 = 60
PC_channel_2 = 60
PC_channel_3 = 60
temperature = 50 #from 10 to 70degC

#END USER INPUT


print("configure DAC6, DAC3-DAC0")
os.system("mcp2210cli -spitxfer=28,4f -bd=100000, -cs=gp4 -md=1")       # configure DAC6, DAC3-DAC0

voltage("DAC0", PC_channel_0)
voltage("DAC1", PC_channel_1)
voltage("DAC2", PC_channel_2)
voltage("DAC3", PC_channel_3)

temp_set(temperature)
