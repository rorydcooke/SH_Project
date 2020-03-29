"""   -----------------Migration_Rate.py--------------------

A script with necessary python pieces to read in data for computing migration. Data is read in and then
stars sorted into initial radii bins. Stars in each bin are then tracked to their location at the
next timestep and probability density histograms constaructed. Either a single or double Gaussian function
 is fitted to the migration curves and then these curves and corresponding fitted functions are plotted.
Finally the sum of the square residuals is calculated for each fit in order to quantify the 'goodness of fit'.

Author: Rory Cooke 23/3/2020



"""



# regular import
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize






def read_migration(infile,norbits=1000000):
    """
    read in binary files of computed orbit migration using memmap

    inputs
    ---------
    infile   : str
        name of the file to be read
    norbits  : int
        number of orbits in the file. default is 1M -- don't change this unless specified.

    returns
    ----------
    apoR     : array of floats
        the apocentre (largest radius) extent of an orbit
    apoV     : array of floats
        the velocity at apocentre
    apoZ     : array of floats
        the maximum height a star reaches

    """
    # define prefactors
    radius_scale = 300. # convert to kpc
    velocity_scale = (240./1.4) # convert to km/s
    #
    apoR = radius_scale*   np.memmap(infile,dtype=np.float32,shape=(norbits))
    apoV = velocity_scale* np.memmap(infile,dtype=np.float32,shape=(norbits),offset=norbits*4)
    apoZ = radius_scale*   np.memmap(infile,dtype=np.float32,shape=(norbits),offset=2*(norbits*4))
    return apoR,apoV,apoZ




timesteps = [0.,1.,2.,3.] # times, in Gyr
# each of the datasets read in below corresponds to a timestep,
# e.g. migration0 corresponds to timesteps[0], or 0. Gyr (the
# beginning of the simulation)

# read in the data
apoR0,apoV0,apoZ0 = read_migration('migration0.dat')
apoR1,apoV1,apoZ1 = read_migration('migration1.dat')
apoR2,apoV2,apoZ2 = read_migration('migration2.dat')
apoR3,apoV3,apoZ3 = read_migration('migration3.dat')
mode=input("To use progressive increment timesteps type 'y':")   #use either progressive t0-t1, t0-t2 timesteps, or sucsessive t0-t1, t1-t2
fit=input("To fit two Gaussian's type 'y'(Default is one Gaussian):")   #fit either a single or double gaussian functions
apolist=[apoR0,apoR1,apoR2,apoR3]
if mode in["y","Y"]:   # open output file to write out the sum of the square residuals for each fit (chi = sum of square residuals), for use in Plot.py
    if fit in["y","Y"]:
        output= open("Chi_prog_2_G.txt", "w")
    else:
        output= open("Chi_prog_1_G.txt","w")
else:
    if fit in["y","Y"]:
        output= open("Chi_suc_2_G.txt", "w")
    else:
        output= open("Chi_suc_1_G.txt","w")

if mode in["y","Y"]:  # open output file to write out fitted paramaters (for use in Metallcity_Gradient)
    if fit in["y","Y"]:
        output_param= open("Params_prog_2_G.txt", "w")
    else:
        output_param= open("Params_prog_1_G.txt","w")
else:
    if fit in["y","Y"]:
        output_param= open("Params_suc_2_G.txt", "w")
    else:
        output_param= open("Params_suc_1_G.txt","w")

#Define both a single and double Gaussian function to be fitted.

def gaussian(x, amp, cen, wid):
    return amp*(1/(wid*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen)/wid)**2)))

def gaussian2(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2)))

gp3=[0.05,2.1,1,0.01,7,4]
gp5=[0.9,0.75,0.1,0.04,0.5,0.05]
gp11=[0.4,4,1,0.6,7,2]
gp15=[0,10,0.3,0.5,11.6,1]
gp17=[0,15,0,0.04,16,0.2]
gp18=[-84,-30,-6,-1,20,-1]
gp19=[-84,-35,-6,-1,19,-0.7]
gp20=[0.3,25,1.5,1,27,4]
gp51=[1,2,0.4,0.4,1,9]
gp111=[0.5,8,1,0.5,5,0.8]
param_bounds=([0,0,0,0,0,0],[30,30,10,30,30,10])   # ensures fitted parameters are not negative and unphysical
if fit in ["Y","y"]:  # keeps a list of guess parameters for double gaussian used to help curve_fit have one for each timestep
    guess_list=[gp5,gp5,gp5,gp3,gp3,gp11,gp11,gp11,gp11,gp11,gp11,gp15,gp15,gp15,gp15,gp17,gp17,gp18,gp19,gp20]
    guess_list1=[gp5,gp5,gp5,gp51,gp51,gp11,gp111,gp111,gp11,gp11,gp11,gp15,gp15,gp15,gp15,gp17,gp17,gp18,gp19,gp20]
    guess_list2=[gp5,gp5,gp5,gp51,gp51,gp11,gp11,gp11,gp11,gp11,gp11,gp15,gp15,gp15,gp15,gp17,gp17,gp18,gp19,gp20]

else:  # keeps a list of guess parameters for single gaussian used to help curve_fit have one for each timestep
    guess_list=[gp5[:-3],gp5[:-3],gp5[:-3],gp5[:-3],gp5[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp15[:-3],gp15[:-3],gp15[:-3],gp15[:-3],gp17,gp17,gp18,gp19,gp20]
    guess_list1=[gp5[:-3],gp5[:-3],gp5[:-3],gp51[:-3],gp5[:-3],gp11[:-3],gp111[:-3],gp111[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp15[:-3],gp15[:-3],gp15[:-3],gp15[:-3],gp17,gp17,gp18,gp19,gp20]
    guess_list2=[gp5[:-3],gp5[:-3],gp5[:-3],gp5[:-3],gp5[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp11[:-3],gp15[:-3],gp15[:-3],gp15[:-3],gp15[:-3],gp17,gp17,gp18,gp19,gp20]

bigguess_list=[guess_list,guess_list1,guess_list2]

# List of titles to be used when plotting data
titles=["P(R_Initial[0>Kpc,<1Kpc]) vs Final Radius(Kpc)","P(R_Initial[1>Kpc,<2Kpc]) vs Final Radius(Kpc)","P(R_Initial[2>Kpc,<3Kpc]) vs Final Radius(Kpc)","P(R_Initial[3>Kpc,<4Kpc]) vs Final Radius(Kpc)","P(R_Initial[4>Kpc,<5Kpc]) vs Final Radius(Kpc)","P(R_Initial[5>Kpc,<6Kpc]) vs Final Radius(Kpc)","P(R_Initial[6>Kpc,<7Kpc]) vs Final Radius(Kpc)","P(R_Initial[7>Kpc,<8Kpc]) vs Final Radius(Kpc)","P(R_Initial[8>Kpc,<9Kpc]) vs Final Radius(Kpc)","P(R_Initial[9>Kpc,<10Kpc]) vs Final Radius(Kpc)","P(R_Initial[10>Kpc,<11Kpc]) vs Final Radius(Kpc)","P(R_Initial[11>Kpc,<12Kpc]) vs Final Radius(Kpc)","P(R_Initial[12>Kpc,<13Kpc]) vs Final Radius(Kpc)","P(R_Initial[13>Kpc,<14Kpc]) vs Final Radius(Kpc)","P(R_Initial[14>Kpc,<15Kpc]) vs Final Radius(Kpc)","P(R_Initial[15>Kpc,<16Kpc]) vs Final Radius(Kpc)","P(R_Initial[16>Kpc,<17Kpc]) vs Final Radius(Kpc)","P(R_Initial[17>Kpc,<20Kpc]) vs Final Radius(Kpc)","P(R_Initial[20>Kpc,<23Kpc]) vs Final Radius(Kpc)","P(R_Initial[23>Kpc,<53Kpc]) vs Final Radius(Kpc)"]
bins=[]
index=[]
bins1=[]
index1=[]
bins2=[]
index2=[]
bins3=[]
index3=[]
bigbin=[bins,bins1,bins2,bins3]
bigindex=[index,index1,index2,index3]


if mode in ["y","Y"]:      # sorts stars into intial 1kpc radii bins
    for i in range(17):
        minv=abs(i)
        maxv= minv +1
        x= apoR0[np.where((apoR0>minv) & (apoR0<maxv))]
        bins.append(x)
        y=np.where((apoR0>minv) & (apoR0<maxv))[0]
        index.append(y)

    x1= apoR0[np.where((apoR0>17) & (apoR0<20))]  #in outer bins where there are fewer stars, larger bins are made to ensure the migration rates are 'smooth' curves.
    bins.append(x1)
    y1=np.where((apoR0>17) & (apoR0<20))[0]
    index.append(y1)
    x2= apoR0[np.where((apoR0>20) & (apoR0<23))]
    bins.append(x2)
    y2=np.where((apoR0>20) & (apoR0<23))[0]
    index.append(y2)
    x3= apoR0[np.where((apoR0>23) & (apoR0<max(apoR0)))]
    bins.append(x3)
    y3=np.where((apoR0>23) & (apoR0<max(apoR0)))[0]
    index.append(y3)

else: # same as above except since sucsessive timesteps are used, need to be able to bin stars in any bin not just apoR0.
    for j in range(3):       #stars initial radii and index are recorded and stored for every timestep
        inr=apolist[j]
        rbin=bigbin[j]
        rindex=bigindex[j]
        for i in range(17):
            minv=abs(i)
            maxv= minv +1
            x= inr[np.where((inr>minv) & (inr<maxv))]
            rbin.append(x)
            y=np.where((inr>minv) & (inr<maxv))[0]
            rindex.append(y)

        x1= inr[np.where((inr>17) & (inr<20))]
        rbin.append(x1)
        y1=np.where((inr>17) & (inr<20))[0]
        rindex.append(y1)
        x2= inr[np.where((inr>20) & (inr<23))]
        rbin.append(x2)
        y2=np.where((inr>20) & (inr<23))[0]
        rindex.append(y2)
        x3= inr[np.where((inr>23) & (inr<max(inr)))]
        rbin.append(x3)
        y3=np.where((inr>23) & (inr<max(inr)))[0]
        rindex.append(y3)


list1=[]
list2=[]
list3=[]
listlist=[list1,list2,list3]
if mode in ["y","Y"]:  # usig the indicies from above stars can be tracked to their new position at every timestep.
    for j in range(len(index)):
        newposT1=np.zeros(bins[j].size)
        newposT2=np.zeros(bins[j].size)
        newposT3=np.zeros(bins[j].size)
        for i in range(bins[j].size):
            indexlist=index[j]      #select bin
            x=indexlist[i]          #select star
            newpos1=(apoR1[x])      #track to new final radii at each timestep
            newpos2=(apoR2[x])
            newpos3=(apoR3[x])
            newposT1[i]=(newpos1)
            newposT2[i]=(newpos2)
            newposT3[i]=(newpos3)
        list1.append(newposT1)     #append these to a list to store them
        list2.append(newposT2)
        list3.append(newposT3)

else:
    for k in range(3):   # as above except data is first selcted from a timetstep list so migration rates can be determined for sucsessive timesteps
        rbin=bigbin[k]
        rindex=bigindex[k]
        inr=apolist[k]
        fr=apolist[k+1]
        corlist=listlist[k]
        for j in range(len(rindex)):
            newpos=np.zeros(rbin[j].size)
            for i in range(rbin[j].size):
                indexlist=rindex[j]
                x=indexlist[i]
                newpos1=(fr[x])
                newpos[i]=(newpos1)
            corlist.append(newpos)

'''These lists are used to store the PDF histogram data. First the probability density
'hist1,2,3 is stored and the the corresponding radii. Either a single or double Gaussian
is then fitted. The corresponding fitted parameters and covariance matricies stored. '''

histlist1=[]
histlist2=[]
histlist3=[]
binslist1=[]
binslist2=[]
binslist3=[]
paramlist1=[]
paramlist2=[]
paramlist3=[]
covmax1=[]
covmax2=[]
covmax3=[]
bighistlist=[histlist1,histlist2,histlist3]
bigbinslist=[binslist1,binslist2,binslist3]
bigparamlist=[paramlist1,paramlist2,paramlist3]
bigcovmax=[covmax1,covmax2,covmax3]
if mode in ["y","Y"]:
    for i in range(15):    #only have accurate rates for first 15 bins. Also when we come to use these rates in Metallicity_Gradient.py it is outside the region of interest.
        in_param=guess_list[i]
        hist1,bins1=np.histogram(list1[i],bins=200,density=True)   #calculate probability density functions for a star finding itself at some final radii x given it started between n-m kpc.
        hist2,bins2=np.histogram(list2[i],bins=200,density=True)
        hist3,bins3=np.histogram(list3[i],bins=200,density=True)
        if fit in ["y","Y"]:
            param1,param_cov1= optimize.curve_fit(gaussian2,bins1[:-1],hist1, p0=in_param,bounds=param_bounds)   #find best fit paramters for 2 Gaussian fit using initial best guesses.
            param2,param_cov2= optimize.curve_fit(gaussian2,bins2[:-1],hist2, p0=in_param,bounds=param_bounds)
            param3,param_cov3= optimize.curve_fit(gaussian2,bins3[:-1],hist3, p0=in_param,bounds=param_bounds)
            histlist1.append(hist1)
            binslist1.append(bins1)
            histlist2.append(hist2)
            binslist2.append(bins2)
            histlist3.append(hist3)
            binslist3.append(bins3)
            paramlist1.append(param1)
            covmax1.append(param_cov1)
            paramlist2.append(param2)
            covmax2.append(param_cov2)
            paramlist3.append(param3)
            covmax3.append(param_cov3)
        else:
            param1,param_cov1= optimize.curve_fit(gaussian,bins1[:-1],hist1, p0=in_param)        #find best fit paramters for single Gaussian fit using initial best guesses.
            param2,param_cov2= optimize.curve_fit(gaussian,bins2[:-1],hist2, p0=in_param)
            param3,param_cov3= optimize.curve_fit(gaussian,bins3[:-1],hist3, p0=in_param)
            histlist1.append(hist1)
            binslist1.append(bins1)
            histlist2.append(hist2)
            binslist2.append(bins2)
            histlist3.append(hist3)
            binslist3.append(bins3)
            paramlist1.append(param1)
            covmax1.append(param_cov1)
            paramlist2.append(param2)
            covmax2.append(param_cov2)
            paramlist3.append(param3)
            covmax3.append(param_cov3)

else:    # identical as above except performed on sucsessive timesteps. First the timestep is selected then the radius bin then the corresponding histogram and parameterisation.
    for j in range(3):
        corlist=listlist[j]
        corhistlist=bighistlist[j]
        corbinslist=bigbinslist[j]
        corparamlist=bigparamlist[j]
        corcovmax=bigcovmax[j]
        corguess_list=bigguess_list[j]
        for i in range(15):
            in_param=corguess_list[i]
            hist1,bins1=np.histogram(corlist[i],bins=200,density=True)
            if fit in ["y","Y"]:
                param1,param_cov1= optimize.curve_fit(gaussian2,bins1[:-1],hist1, p0=in_param,maxfev=100000,bounds=param_bounds)
                corhistlist.append(hist1)
                corbinslist.append(bins1)
                corparamlist.append(param1)
                corcovmax.append(param_cov1)
            else:
                param1,param_cov1= optimize.curve_fit(gaussian,bins1[:-1],hist1, p0=in_param,maxfev=100000,bounds=([0,0,0],[30,30,10]))
                corhistlist.append(hist1)
                corbinslist.append(bins1)
                corparamlist.append(param1)
                corcovmax.append(param_cov1)


def plotter(x1,y1,x2,y2,x3,y3,p1,p2,p3,titles,co1,co2,co3):   #plots relevant data.

    if mode in["y","Y"]:   #use either sucsessivce or progressive timesteps
        for i in range(15):   #plot required range
            if bins[i].size==0:   # ensures empty bins are not plotted.
                pass
            else:
                val=abs(i)
                title=titles[val]
                x1x=x1[i]  # x data for bin(val) t0-t1
                x2x=x2[i]
                x3x=x3[i]
                y1y=y1[i]
                y2y=y2[i]
                y3y=y3[i]  # y data for bin(val) t0-t3
                p1p=p1[i]
                p2p=p2[i]  #fitted parameter list
                p3p=p3[i]
                plt.plot(x1x[:-1],y1y,label="T0-T1")
                plt.plot(x2x[:-1],y2y,label="T0-T2")
                plt.plot(x3x[:-1],y3y,label="T0-T3")  #plots PDF histograms
                plt.title(title)
                plt.xlabel("Final Radius (Kpc)")
                plt.ylabel("P(R_Initial)")
                #plt.legend(loc="upper right")
                if fit in["y","Y"]:
                    plt.plot(x1x[:-1],gaussian2(x1x[:-1],*p1p),label="Two Gaussian Fit (T0-T1)")   #plots fitted functions
                    plt.plot(x2x[:-1],gaussian2(x2x[:-1],*p2p),label="Two Gaussian Fit (T0-T2)")
                    plt.plot(x3x[:-1],gaussian2(x3x[:-1],*p3p),label="Two Gaussian Fit (T0-T3)")
                    #used to save figures
                    # plt.legend(loc="upper right")
                    # plt.savefig("/home/s1609274/SHProject/Data/Figures/Double_Prog/"+str(val)+".png")
                    # plt.close()
                    # write out parameters to file
                    output_param.write("Parameters T0-T1:  "+str(p1p[0]) +' , ' +str(p1p[1]) +' , ' +str(p1p[2])+' , '+str(p1p[3]) +' , ' +str(p1p[4]) +' , ' +str(p1p[5])+' \n '
                    "Parameters T0-T2:  "+str(p2p[0]) +' , ' +str(p2p[1]) +' , ' +str(p2p[2])+' , '+str(p2p[3]) +' , ' +str(p2p[4]) +' , ' +str(p2p[5])+' \n '
                    "Parameters T0-T3:  "+str(p3p[0]) +' , ' +str(p3p[1]) +' , ' +str(p3p[2])+' , '+str(p3p[3]) +' , ' +str(p3p[4]) +' , ' +str(p3p[5])+' \n ')

                else:
                    plt.plot(x1x[:-1],gaussian(x1x[:-1],*p1p),label="Single Gaussian Fit (T0-T1)")
                    plt.plot(x2x[:-1],gaussian(x2x[:-1],*p2p),label="Single Gaussian Fit (T0-T2)")
                    plt.plot(x3x[:-1],gaussian(x3x[:-1],*p3p),label="Single Gaussian Fit (T0-T3)")
                    # Uncomment to save figures instead of plot them
                    # plt.legend(loc="upper right")
                    # plt.savefig("/home/s1609274/SHProject/Data/Figures/Single_Prog/"+str(val)+".png")
                    # plt.close()                    # write out parameters to file
                    output_param.write("Parameters T0-T1:  "+str(p1p[0]) +' , ' +str(p1p[1]) +' , ' +str(p1p[2])+' \n '
                    "Parameters T0-T2:  "+str(p2p[0]) +' , ' +str(p2p[1]) +' , ' +str(p2p[2])+' \n '
                    "Parameters T0-T3:  "+str(p3p[0]) +' , ' +str(p3p[1]) +' , ' +str(p3p[2])+' \n ')
                plt.title(title)
                plt.xlabel("Final Radius (Kpc)")
                plt.ylabel("P(R_Initial)")
                plt.legend(loc="upper right")
                plt.show()
        return()

    else:
        for i in range(15):
            if bins[i].size==0:
                pass
            else:
                val=abs(i)
                title=titles[val]
                x1x=x1[i]
                x2x=x2[i]
                x3x=x3[i]
                y1y=y1[i]
                y2y=y2[i]
                y3y=y3[i]
                p1p=p1[i]
                p2p=p2[i]
                p3p=p3[i]
                plt.plot(x1x[:-1],y1y,label="T0-T1")
                plt.plot(x2x[:-1],y2y,label="T1-T2")
                plt.plot(x3x[:-1],y3y,label="T2-T3")
                # uncomment if you want to svae figures instead of plot
                # plt.title(title)
                # plt.xlabel("Final Radius (Kpc)")
                # plt.ylabel("P(R_Initial)")
                # plt.legend(loc="upper right")
                if fit in["y","Y"]:
                    plt.plot(x1x[:-1],gaussian2(x1x[:-1],*p1p),label="Two Gaussian Fit (T0-T1)")
                    plt.plot(x2x[:-1],gaussian2(x2x[:-1],*p2p),label="Two Gaussian Fit (T1-T2)")
                    plt.plot(x3x[:-1],gaussian2(x3x[:-1],*p3p),label="Two Gaussian Fit (T2-T3)")
                    # plt.legend(loc="upper right")
                    # plt.savefig("/home/s1609274/SHProject/Data/Figures/Double_Suc/"+str(val)+".png")
                    # plt.close()
                    # output in CSV format so data can be read in easily for Metallicity_Gradient.py
                    output_param.write(str(p1p[0])+" ,  "+str(p1p[1])+" ,  "+str(p1p[2])+" ,  "+str(p1p[3])+" ,  "+str(p1p[4])+" ,  "+str(p1p[5])+"  ,   "+" 0 "+"  , "+str(val)+"\n"
                    +str(p2p[0])+" ,  "+str(p2p[1])+" ,  "+str(p2p[2])+" ,  "+str(p2p[3])+" ,  "+str(p2p[4])+" ,  "+str(p2p[5])+"  ,  "+" 1 "+"  , "+str(val)+"\n"
                    +str(p3p[0])+" ,  "+str(p3p[1])+" ,  "+str(p3p[2])+" ,  "+str(p3p[3])+" ,  "+str(p3p[4])+" ,  "+str(p3p[5])+" ,  "+" 2 "+"  , "+str(val)+"\n")
                else:
                    plt.plot(x1x[:-1],gaussian(x1x[:-1],*p1p),label="Single Gaussian Fit (T0-T1)")
                    plt.plot(x2x[:-1],gaussian(x2x[:-1],*p2p),label="Single Gaussian Fit (T1-T2)")
                    plt.plot(x3x[:-1],gaussian(x3x[:-1],*p3p),label="Single Gaussian Fit (T2-T3)")
                    # uncomment to save figures instead of plot
                    # plt.legend(loc="upper right")
                    # plt.savefig("/home/s1609274/SHProject/Data/Figures/Single_Suc/"+str(val)+".png")
                    # plt.close()
                    output_param.write("Parameters T0-T1:  "+str(p1p[0]) +' , ' +str(p1p[1]) +' , ' +str(p1p[2])+' \n '
                    "Parameters T1-T2:  "+str(p2p[0]) +' , ' +str(p2p[1]) +' , ' +str(p2p[2])+' \n '
                    "Parameters T2-T3:  "+str(p3p[0]) +' , ' +str(p3p[1]) +' , ' +str(p3p[2])+' \n ')
                plt.title(title)
                plt.xlabel("Final Radius (Kpc)")
                plt.ylabel("P(R_Initial)")
                plt.legend(loc="upper right")
                plt.show()
        return()

plotter(binslist1,histlist1,binslist2,histlist2,binslist3,histlist3,paramlist1,paramlist2,paramlist3,titles,covmax1,covmax2,covmax3)



def chi_sqrd(x1,y1,x2,y2,x3,y3,p1,p2,p3,co1,co2,co3   # method to calculate the sum of the square residuals for every fit in every radius bin for every timestep.
    for i in range(15):
        if bins[i].size==0:
            pass
        else:
            val=abs(i)
            title=titles[val]
            x1x=x1[i]
            x2x=x2[i]
            x3x=x3[i]
            y1y=y1[i]
            y2y=y2[i]
            y3y=y3[i]
            p1p=p1[i]
            p2p=p2[i]
            p3p=p3[i]
            if fit in["y","Y"]:
                fittedfx1=gaussian2(x1x[:-1],*p1p)  #calculate 2 gaussian fitted functions for a given timetstep
                fittedfx2=gaussian2(x2x[:-1],*p2p)
                fittedfx3=gaussian2(x3x[:-1],*p3p)
            else:
                fittedfx1=gaussian(x1x[:-1],*p1p)    #calculate single gaussian fitted functions for a given timetstep
                fittedfx2=gaussian(x2x[:-1],*p2p)
                fittedfx3=gaussian(x3x[:-1],*p3p)
            arrchisq1=(fittedfx1-y1y)**2   #subtract from real data from fitted function and square
            arrchisq2=(fittedfx2-y2y)**2
            arrchisq3=(fittedfx3-y3y)**2
            chisq1=np.sum(arrchisq1)   # sum the square differences
            chisq2=np.sum(arrchisq2)
            chisq3=np.sum(arrchisq3)
            #print(chisq1,chisq2,chisq3)
            if fit in["y","Y"]:   #write out to file for use in plot.py
                if mode in["y","Y"]:
                    #output.write("Chi-Squared Statistic for Two Gaussian Fit(T0-T1)[Radius Bin"+str(val)+"]: "+str(chisq1) +" (T0-T2): " +str(chisq2) +" (T0-T3): " +str(chisq3)+'\n')
                    output.write(str(chisq1) +'\n' +str(chisq2) +'\n' +str(chisq3)+'\n')
                else:
                    #output.write("Chi-Squared Statistic for Two Gaussian Fit(T0-T1)[Radius Bin"+str(val)+"]: "+str(chisq1) +" T1-T2: " +str(chisq2) +" T2-T3: " +str(chisq3)+'\n')
                    output.write(str(chisq1) +'\n' +str(chisq2) +'\n' +str(chisq3)+'\n')
            else:
                if mode in["y","Y"]:
                    #output.write("Chi-Squared Statistic for Single Gaussian Fit(T0-T1)[Radius Bin"+str(val)+"]: "+str(chisq1) +" (T0-T2): " +str(chisq2) +" (T0-T3): " +str(chisq3)+'\n')
                    output.write(str(chisq1) +'\n' +str(chisq2) +'\n' +str(chisq3)+'\n')
                else:
                    #output.write("Chi-Squared Statistic for Single Gaussian Fit(T0-T1)[Radius Bin"+str(val)+"]: "+str(chisq1) +" (T1-T2): " +str(chisq2) +" (T2-T3): " +str(chisq3)+'\n')
                    output.write(str(chisq1) +'\n' +str(chisq2) +'\n' +str(chisq3)+'\n')


chi_sqrd(binslist1,histlist1,binslist2,histlist2,binslist3,histlist3,paramlist1,paramlist2,paramlist3,covmax1,covmax2,covmax3)
