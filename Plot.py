'''---------------plot.py----------------
Code with method to plot the ratio of the sum of the square residuals (chi),
for required timesteps over specified 0-15Kpc range.

Author: Rory Cooke 23/3/202'''

import numpy as np
import matplotlib.pyplot as plt


def main():



    filein=open("Output.txt","r")
    filein1=open("Output1.txt","r")
    filein2=open("Output2.txt","r")
    filein3=open("Output3.txt","r")
    data=[]
    data1=[]
    data2=[]
    data3=[]
    x01=[]
    x02=[]
    x03=[]
    x12=[]
    x23=[]
    y=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]



    for line in filein.readlines():    # read in 'chi' from input files, append to list.
        if not line.startswith("#"):
            data.append(float(line))
    filein.close()

    for line in filein1.readlines():
        if not line.startswith("#"):
            data1.append(float(line))
    filein1.close()

    for line in filein2.readlines():
        if not line.startswith("#"):
            data2.append(float(line))
    filein2.close()

    for line in filein3.readlines():
        if not line.startswith("#"):
            data3.append(float(line))
    filein3.close()

    thirdlen=(len(data)/3)
    for i in range(thirdlen):  #calculate ratio of chi for every timestep (both sucsessive and progressive)
        g20=data[(3*i)]
        g21=data[(3*i)+1]
        g22=data[(3*i)+2]
        g10=data1[(3*i)]
        g11=data1[(3*i)+1]
        g12=data1[(3*i)+2]
        g212=data2[(3*i)+1]
        g223=data2[(3*i)+2]
        g112=data3[(3*i)+1]
        g123=data3[(3*i)+2]
        r01=g10/g20
        r02=g11/g212
        r03=g12/g22
        r12=g112/g212
        r23=g123/g223
        x01.append(r01)
        x02.append(r02)
        x03.append(r03)
        x12.append(r12)
        x23.append(r23)

    def plot(x01,x02,x03,x12,x23,y):   #plot the ratios
        plt.plot(y,x01,label="T01")
        plt.plot(y,x02,label="T02")
        plt.plot(y,x03,label="T03")
        plt.plot(y,x12,label="T12")
        plt.plot(y,x23,label="T23")
        plt.legend(loc="upper left")
        plt.xlabel("Initial Radius(KPc)")
        plt.ylabel("Chi Squared Ratio(Single Fit/Double Fit)")
        plt.show()



    plot(x01,x02,x03,x12,x23,y)

main()
