import os
import math
def voltage(channel, V_out):
    V_out = float(V_out)
    dac_code = round(V_out*4095/(5.088*25.877))     # convert desired voltage to hex code
    dac_code_hex = "{:03x}".format(dac_code)        # format code to always look like "fff"
    dac_code_str = str(dac_code_hex)

    dac=str(hex(8+int(channel[3])))                 # convert DAC number to SPI DAC adress (eg. DAC0 -> 0x8)    
    dac_num=dac[2]
    
    terminal = "mcp2210cli -spitxfer=" + dac_num + dac_code_str[0] + "," + dac_code_str[1:] + " -bd=100000, -cs=gp4 -md=1"
    print(terminal)             # print executed SPI command in terminal
    os.system(terminal)         # execute SPI command


# set desired approximate temperature
def temp_set(T_th):
    T_th=float(T_th)
    # V_th=(T_th-90.0)/((65.0-40.0)/(0.877-1.762))                          # approximate thermistor voltage
    R_th0=10e3
    Beta=3950.0
    TempK=273.0+T_th
    T0=298
    V_th=5.088*(R_th0*math.exp(Beta*(1/TempK-1/T0)))/(10e3+R_th0*math.exp(Beta*(1/TempK-1/T0)))
    V_dac=(V_th-0.632)/0.584        # V_th=0.569*V_dac+0.743                # voltage to send to DAC
    print("Thermistor voltage: ~", round(V_th, ndigits=2), "V", sep="")     # thermistor voltage
    dac_code = round(V_dac*4095/5.088)                                      # decimal code to send to dac
    dac_code_hex = "{:03x}".format(dac_code)                                # hexadecimal code to send to dac
    dac_code_str = str(dac_code_hex)                                        # convert code to string to form SPI command
    terminal = "mcp2210cli -spitxfer=E" + dac_code_str[0] + "," + dac_code_str[1:] + " -bd=100000, -cs=gp4 -md=1"
    print(terminal)             # print executed SPI command in terminal
    os.system(terminal)         # execute SPI command


# convert read hexadecimal value from ADC4 to approximate temperature
def temp():
    print("\nTemperature measurement")
    os.system("mcp2210cli -spitxfer=20,10 -bd=100000, -cs=gp4 -md=1")       # SPI command: Define ADC4 as input
    os.system("mcp2210cli -spitxfer=12,10 -bd=100000, -cs=gp4 -md=1")       # SPI command: Push ADC4 data repetitively
    print("input recieved hex input in folowing format '0xfff' (take last 3 hex numbers from RxData))")
    t_read=input()                                  # input ADC value to terminal
    t_read_int=int(t_read, 16)                      # convert to decimal value
    V_th=t_read_int*5.088/4095                      # convert into thermistor voltage
    print("Thermistor voltage: ~", round(V_th, ndigits=2), "V", sep="")
    R_th0=10e3
    R_th=(R_th0*V_th)/(5.088-V_th)                   # calculate approximate thermistor temperature
    Beta=3950
    T0=298          # 25deg in Kelvins
    T_th=1/(math.log(R_th/R_th0)/Beta+1/T0)-273
    print("~", round(T_th, ndigits=2), "deg")
    print("end temperature measurement")


# convert read hexadecimal value from ADC4 to approximate temperature
def temp_approximate():
    print("Temperature measurement")
    os.system("mcp2210cli -spitxfer=20,10 -bd=100000, -cs=gp4 -md=1")       # SPI command: Define ADC4 as input
    os.system("mcp2210cli -spitxfer=12,10 -bd=100000, -cs=gp4 -md=1")       # SPI command: Push ADC4 data repetitively
    print("input recieved hex input in folowing format '0xfff' (take last 3 hex numbers))")
    t_read=input()                                  # input ADC value to terminal
    t_read_int=int(t_read, 16)                      # convert to decimal value
    V_th=t_read_int*5.088/4095                      # convert into thermistor voltage
    print(round(V_th, ndigits=2))
    T_th=90+(65-40)/(0.877-1.762)*V_th              # calculate approximate thermistor temperature
    print("~", round(T_th, ndigits=1), "deg")


# output rising voltage on desired DAC
def ramp(channel, start=0, step=15, stop=4095):                      # takes "DAC0", "DAC1", "DAC2" or "DAC3" as an input string
    dac=str(hex(8+int(channel[3])))     # convert DAC number to SPI DAC adress (eg. DAC0 -> 0x8)
    dac_num=dac[2]                         
    i=start
    while i < (stop+1):
        sest = "{:03x}".format(i)       # format code to alway look like "fff"
        sest1=sest[0]                   # extract part of hex number
        sest2=sest[1:]                  # extract part of hex number
        print(sest1, sest2, sep=",")    # print separated hex values with separator ','
        terminal = "mcp2210cli -spitxfer=" + dac_num + sest1 + "," + sest2 + " -bd=100000, -cs=gp4 -md=1"
        print(terminal)                 # print executed SPI command in terminal
        os.system(terminal)             # execute SPI command
        i=i + step                        # step of rising voltage
