'''----------Metallcity_Gradient.py-------------------------------------------------------
This code aims to invert the method used in N.Frankel et al. 2018: Measuring Radial Orbital Migration in the Galactic Disk, by taking their
best fit parameters and evolving a user specified number of stars according to said parameters. The radial migration aspect of the model
has been substituted for  empirically determined radial migration rate taken from Migration_Rate.py.

 Auhtor: Rory Cooke 23/03/2020'''

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import *
from scipy import stats
from scipy.interpolate import interp1d
import sys
from scipy.stats import kde

sys.setrecursionlimit(1500)

#Adopted all aspects of the model used in N.Frankel et al. 2018: Measuring Radial Orbital Migration in the Galactic Disk (NF1)
#Used all of their fitted model parameters, but used own empirically determined radial migration rate taken from migration.py
# Author: Rory Cooke

t_max=12.    # this is the max age of the disk taken to be 12Gyr
t_sfr= 6.77  # measured in Gyrs, model parameter setting star formation history
delta_Fe_H= -0.075 # ISM metallicity gradient measured in dex/kpc,
F_m= -1. #minimum metallicity at disk centre, measured in dex
R_metal_now= 8.74 # radius at which present day metallicity is solar [Fe/H]=0
gamma_metal= 0.32 #power law exponent governing time dependecy of chemical enrichment
#params=[0.25,2.1,1,0.1,7,2]  #test paramaters (each radius bin has different params for a given t, which should i use?)
sig=3.6
params=np.loadtxt("Params_suc_2_G.txt",delimiter=',')  #Emperically determined radial migration rate taken from migration.py





def stars_at_step(tauvals,t_max,t_sfr,n):  #method returning the number of new stars formed at each timestep and the SFH fraction
    SFH=np.exp(-(t_max-tauvals)/t_sfr)   #functional form taken from NF1
    new_stars=np.array([int(np.round(x,0))for x in(n*(SFH)/np.nansum(SFH))])
    return(new_stars,SFH)

def get_metal(F_m,delta_Fe_H,R_metal_now,gamma_metal,t_max,R,tau):  # method returning metallicity at a given radius and time
    f_t=(1-(tau/t_max))**gamma_metal
    Fe_H=((delta_Fe_H*R)+F_m)-((F_m+delta_Fe_H*R_metal_now)*f_t)  #functional form taken from NF1
    return(Fe_H)


def radial_mig(r,amp1,cen1,sigma1,amp2,cen2,sigma2): # method returning a stars new galactocentric radius
    return(amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((r-cen1)/sigma1)**2))) + \
    amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((r-cen2)/sigma2)**2))))   #Emperically determined radial migration rate taken from migration.py


def radial_mig_diff(r,amp1,cen1,sigma1): # dummy function for stars with R>15kpc
    return(r)   # stars do not migrate

def rad_mig_acc_rej(r,r_0,params,mode): #method that migrates all stars using the empirical relations derived from whatever timestep you are at to the next,
                                        #based on the radii of the stars. Build acceptance-rejection based on the cumulative distribution of curves.
    if mode !=0:   # ensures stars with R>15Kpc are not migrated
        loop_count=0
        star_in_bin=r.size
        while loop_count==0:  #ensures that the number of accepted values = number of stars to be migrated
            dummy_r = np.linspace(0,30,2000)     #dummy target distribution
            enclosed_fraction = np.cumsum(radial_mig(dummy_r,*params))/np.nanmax(np.cumsum(radial_mig(dummy_r,*params)))  #Build acceptance-rejection based on the cumulative distribution of curves
            distvals = interp1d(enclosed_fraction,dummy_r)    # now we need to interpolate the curve at a given time so we can select stars.
            randval = np.random.rand(star_in_bin)     ) # draw n samples randomly from [0,1], where here X=1000000
            valid = (randval > np.nanmin(enclosed_fraction)) & (randval < np.nanmax(enclosed_fraction))    #Values are accpeted or rejected
            rand_new_stars=randval[valid]
            if rand_new_stars.size < r.size:   # ensures that the number of accepted values = number of stars to be migrated
                star_in_bin+=1
            elif rand_new_stars.size > r.size:
                star_in_bin-=1
            else:
                break
        new_rvals=distvals(randval[valid])  # draw accepted values from target distribution
        return(new_rvals)
    else:
        loop_count=0
        star_in_bin=r.size
        while loop_count==0:
            dummy_r = np.linspace(0,30,2000)
            enclosed_fraction = np.cumsum(radial_mig_diff(dummy_r,1,r_0,2))/np.nanmax(np.cumsum(radial_mig_diff(dummy_r,1,r_0,2)))
            distvals = interp1d(enclosed_fraction,dummy_r)
            randval = np.random.rand(star_in_bin)
            valid = (randval > np.nanmin(enclosed_fraction)) & (randval < np.nanmax(enclosed_fraction))
            rand_new_stars=randval[valid]
            if rand_new_stars.size < r.size:
                star_in_bin+=1
            elif rand_new_stars.size > r.size:
                star_in_bin-=1
            else:
                break

        new_rvals=distvals(randval[valid])
        return(new_rvals)

def exponential_at_time(radius,tau,alpha=0.42):  #method governing stars birth radii at time tau
    rexp = 3.*(1.-alpha*(tau/8.))    #functional form taken from NF1
    return np.exp(-radius/rexp)/rexp

def generate_dist(t,new_stars,tauvals):    #method that generates a distribution of stars using the relationstaken from NF1,
                                        #Builds acceptance-rejection based on the cumulative distribution of curves.
    loop_count=0
    ns_counter=new_stars
    while loop_count==0:
        rvals = np.linspace(0.,30.,2000) # in kpc
        enclosed_fraction = np.cumsum(exponential_at_time(rvals,tauvals))/np.nanmax(np.cumsum(exponential_at_time(rvals,tauvals)))
        distvals = interp1d(enclosed_fraction,rvals)
        randval = np.random.rand(ns_counter) # draw new_stars samples randomly from [0,1], where here new_stars=1000000
        valid = (randval > np.nanmin(enclosed_fraction)) & (randval < np.nanmax(enclosed_fraction))
        rand_new_stars=randval[valid]
        if rand_new_stars.size < new_stars:
            ns_counter+=1
        elif rand_new_stars.size > new_stars:
            ns_counter-=1
        else:
            break
    rval=distvals(randval[valid])
    metal_val=get_metal(F_m,delta_Fe_H,R_metal_now,gamma_metal,t_max,rval,tauvals) #assigns a metallicity to the newly generated distribution of stars
    return(rval,metal_val)

def plot_data(x,y,r,mean,median,mean_par,median_par,mean_cov,median_cov):
    #reduce arrays to plot in 6-15kpc range
    x_plot=x[np.where((x>6)&(x<15))]
    y_plot=y[np.where((x>6)&(x<15))]
    r_plot=r[:-1]

    #plot data in required range
    plt.scatter(x_plot,y_plot,s=0.1)
    plt.plot(r_plot,mean,label="Best fit Mean")  #plot the mean metallicity value in each 0.5Kpc bin
    plt.plot(r_plot,median,label="Best fit Median")  #plot the median metallicity value in each 0.5Kpc bin
    plt.plot(r_plot,r_plot*mean_par[0] +mean_par[1],label="Linear Regression Mean")   #plot the linear regression of the mean metallicity value in each bin
    plt.plot(r_plot,r_plot*median_par[0] +median_par[1],label="Linear Regression Median")   #plot the linear regression of the median metallicity value in each bin
    plt.ylabel("Metallicity [Fe/H]")
    plt.xlabel("Radius (Kpc)")
    plt.title("Metallicity [Fe/H] vs Galactocentric Radius (Kpc)")
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([6,15])  #plot required range
    plt.show()

    #plot 2d hexabin (histogramp)
    plt.hexbin(x_plot,y_plot,gridsize=200,cmap='inferno')
    #plt.plot(x_plot,gradient*x_plot +y_int,label="Best Fit Metallicity Gradient dex/Kpc",c='r')
    plt.plot(r_plot,mean,label="Best fit Mean")
    plt.plot(r_plot,median,label="Best fit Median")
    plt.plot(r_plot,r_plot*mean_par[0] +mean_par[1],label="Linear Regression Mean")
    plt.plot(r_plot,r_plot*median_par[0] +median_par[1],label="Linear Regression Median")
    plt.ylabel("Metallicity [Fe/H]")
    plt.xlabel("Galactocentric Radius (Kpc)")
    axes = plt.gca()
    axes.set_xlim([6,14])
    plt.colorbar()
    plt.title("2D Hexbin Plot of Metallicity [Fe/H] vs Galactocentric Radius (Kpc)")
    plt.legend()
    plt.show()

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins=300
    k = kde.gaussian_kde([x_plot,y_plot])
    xi, yi = np.mgrid[x_plot.min():x_plot.max():nbins*1j, y_plot.min():y_plot.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))


    # Change color palette
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.inferno)
    plt.plot(r_plot,mean,label="Best fit Mean")
    plt.plot(r_plot,median,label="Best fit Median")
    plt.plot(r_plot,r_plot*mean_par[0] +mean_par[1],label="Linear Regression Mean")
    plt.plot(r_plot,r_plot*median_par[0] +median_par[1],label="Linear Regression Median")
    plt.colorbar()
    plt.ylabel("Metallicity [Fe/H]")
    plt.xlabel("Galactocentric Radius (Kpc)")
    plt.title("2D Colour Density Plot of Metallicity [Fe/H] vs Galactocentric Radius (Kpc)")
    axes = plt.gca()
    axes.set_xlim([6,14])
    plt.legend()
    plt.show()

    return()

def line(x,m,c):  # functional form of straight line used in linear regression
    return(m*x +c)

def get_metal_grad(R,M):  # method that calculates mean and median values in 0.5Kpc bins
    # define the radial bins
    rbins = np.arange(0.,16.5,0.5)

    # define a median and mean blank array
    median_Z = np.zeros(rbins.size-1)
    mean_Z = np.zeros(rbins.size-1)

    for indr,r in enumerate(rbins[0:-1]):
        w = np.where( (R>rbins[indr]) & (R< rbins[indr+1]))
        median_Z[indr] = np.nanmedian(M[w])
        mean_Z[indr] = np.nanmean(M[w])

    return(median_Z,mean_Z,rbins)

def linear_reg_2(r,med,mean):  # method to calculate linear regresion on the mean and median values
    #print(r.size,med.size,mean.size)
    p_mean,c_mean=optimize.curve_fit(line,r[:-1],mean,p0=[-0.075,1])
    p_med,c_med=optimize.curve_fit(line,r[:-1],med,p0=[-0.075,1])
    return(p_mean,c_mean,p_med,c_med)  # returns best fited parameters and covariance matrices

def linear_reg(r,m): #method to calculate linear regression on all data points
    p,c=optimize.curve_fit(line,r,m,p0=[-0.075,1])
    grad=p[0]
    err=c[0,0]
    return(grad,err)


def main_v2():
    tvals= np.linspace(3.,13,18.)  # timestep array to be iterated over (only have migration rates for every ~3yr but use 18 timestepts to remove discreeteness)
    tauvals=t_max-tvals  # defines 'look back' time as used in NF1
    mean_grad_evol=[]   # empy lits to store the time evolution of the mean/median gradients
    median_grad_evol=[]
    t_evol=[]
    median_err_evol=[]  # empty list to store the time evolution of the errros on the mean/mdian gradients
    mean_err_evol=[]
    mig_strength=[]   # stores the average sigma (strength of migration) for every time step
    n=1000001
    tot_stars,SFH=stars_at_step(tauvals,t_max,t_sfr,n)   # find how many stars should be formed at each step. Return SFH to see if it correlates with time evol. gradient
    rad_arr=np.zeros(n)  # empty arrays that store all stars metallicity and radii. 0s indicate a star yet to be formed.
    metal_arr=np.zeros(n)
    prev_t=0.0
    n_formed=0  # keeps track of the number of new stars formed
    for index,t in enumerate(tvals):
        sig_list=[]   # stores the average sigma for each radial bin ( the mean value of this is given to mig_strength)
        new_stars=tot_stars[index]
        rvals,metal_vals=generate_dist(t,new_stars,tauvals[index])   #generate dist of stars with random radius,and assign them a metallicity at birth
        for i in np.arange(n_formed,n_formed+rvals.size):  # stores all newly generated birth radii in the array of zeros
            rad_arr[i]=rvals[i-n_formed]
        for i in np.arange(n_formed,n_formed+metal_vals.size):
            metal_arr[i]=metal_vals[i-n_formed]
        n_formed += rvals.size    # track how many stars have formed
        migration_radii = np.linspace(0.,30.,31.)
        dradius = migration_radii[1]-migration_radii[0]   # used to migrate stars in 1Kpc bins
        outer_r_params=np.zeros(6)   # parameter for 14-15Kpc bin are used for the reamming radius bins as accurate parameters could not be determined emperically
        for indxr,rad in enumerate(migration_radii):   #parameters are not actually used to migrate stars theya re just there for completeness.
            if rad <15:  # migrates 1kpc bins of all stars with r<15kpc
                if index <6:   # the same set of parameters are used for 6 timesteps (in order to remove discreetness in the final metallicity vs radius plot)
                    w = np.where( (rad_arr > rad) & (rad_arr < rad+dradius))[0] # identify particles in this bin
                    y= np.where((rad==params[:,7]) & (0==params[:,6]))[0]   # identifies the correct radius bin and timestep
                    i=np.array([0,1,2,3,4,5])
                    parameters=params[y,i]  # all 6 parameters required for migration
                    outer_r_params=parameters
                    new_rvals=rad_mig_acc_rej(rad_arr[w],rad,parameters,1)
                    rad_arr[w] = new_rvals
                    x=(parameters[2]+parameters[5])/2   # take average of sigma1 and sigma2
                    sig_list.append(x)

                elif 5<index<12:
                    w = np.where( (rad_arr > rad) & (rad_arr < rad+dradius))[0] # identify particles in this bin
                    y= np.where((rad==params[:,7]) & (1==params[:,6]))[0]
                    i=np.array([0,1,2,3,4,5])
                    parameters=params[y,i]
                    outer_r_params=parameters
                    new_rvals=rad_mig_acc_rej(rad_arr[w],rad,parameters,1)
                    rad_arr[w] = new_rvals
                    x=(parameters[2]+parameters[5])/2
                    sig_list.append(x)
                else:
                    w = np.where( (rad_arr > rad) & (rad_arr < rad+dradius))[0] # identify particles in this bin
                    y= np.where((rad==params[:,7]) & (2==params[:,6]))[0]
                    i=np.array([0,1,2,3,4,5])
                    parameters=params[y,i]
                    outer_r_params=parameters
                    new_rvals=rad_mig_acc_rej(rad_arr[w],rad,parameters,1)
                    rad_arr[w] = new_rvals
                    x=(parameters[2]+parameters[5])/2
                    sig_list.append(x)
            else:
                new_rvals=rad_mig_acc_rej(rad_arr[w],rad,outer_r_params,0)  # the
                rad_arr[w] = new_rvals
                x=(parameters[2]+parameters[5])/2
                sig_list.append(x)

        med,mn,rbin=get_metal_grad(rad_arr,metal_arr)   # get the mean/median values of metallicity
        mean,err_mean,median,err_median=linear_reg(rbin,med,mn)   #calculate linear regression of those values
        mean_grad_evol.append(mean[0]),median_grad_evol.append(median[0]),t_evol.append(t),mean_err_evol.append(err_mean[0,0]),median_err_evol.append(err_median[0,0])  #append to time evolution list


        av_sig=np.mean(sig_list)
        mig_strength.append(av_sig)

    l_median,l_mean,r=get_metal_grad(rad_arr,metal_arr) # get final mean\median metallciities in bin
    f_mean,f_err_mean,f_median,f_err_med=linear_reg_2(r,l_median,l_mean)   #get fianl mean/median gradients and associated errors
    lin_m,lin_m_err=linear_reg(rad_arr,metal_arr)  # get the final gradient

    #Plot the time dependence of parameters

    plt.plot(t_evol,mean_grad_evol,label="Mean")
    plt.plot(t_evol,median_grad_evol,label="Median")
    plt.xlabel("Time (GYrs)")
    plt.ylabel("Metallicity Gradient ([Fe/H]dex/Kpc)")
    plt.title("Time dependence of Metallcity Gradient")
    plt.legend()
    plt.show()
    plt.plot(t_evol,SFH)
    plt.xlabel("Time (GYrs)")
    plt.ylabel("SFH")
    plt.title("Time dependence of SFH")
    plt.show()
    plt.plot(t_evol,mig_strength)
    plt.xlabel("Time (GYrs)")
    plt.ylabel("Mean strength of Migration")
    plt.title("Time dependence of Sigma(Strength of Migration)")
    plt.show()
    '''
    print("The Present Day Mean MW ISM Metallicity Gradient is:  "+str(f_mean[0])+"  with error: "+str(f_err_mean[0,0])+" . The Present Day Median MW ISM Metallicity Gradient is:  "+str(f_median[0])+"  with error: "+str(f_err_mean[0,0]) )
    print("The Present Day MW ISM Metallicity Gradient is:  "+str(lin_m)+"  with error: "+str(lin_m_err))

main_v2()
