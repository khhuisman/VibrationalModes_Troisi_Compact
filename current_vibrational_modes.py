#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman


import numpy as np 
from matplotlib import pyplot as plt
import negf_vibrational_modes as negf_method



def trapz_integrate(xarray,y,*args):
    
    '''
    Returns integral over interval with trapezium method.
    
    --------------
    Input:
    xarray = array of values to integrat over
    y = function
    *args = arguments of function y
    
    Returns
    integral of function y with arguments *args over the interval xarray
    
    '''
    yarray = y(xarray,*args)
    
    return np.trapz(yarray,xarray),yarray

def calc_bound_integrand_current(elist,nround,f,*args):
    
    '''
    Input:
    elist = guess where upper and lower bound of integrand will tend to zero.
    nround = numerical order one wants to neglect
    f = considered function
    args = arguments of that function
    Output:
    lower and upper bound [elower,eupper] outside which the integrand is to the numerical order given by nround.
    '''
    
    
    e_lower = min(elist) - 1
    
    
    
    integrand_lower = abs( np.round( f(np.array([e_lower]),
                                 *args ),nround
                                   )
                         )[0] 
    
    
    
    
    while integrand_lower != 0:
        e_lower -= 0.1
        integrand_lower = abs( np.round( f(np.array([e_lower]),
                                 *args ),nround
                                   )
                         )[0] 
        
        
        
        
    e_upper = max(elist) + 1
    
    integrand_upper  = abs( np.round( f(np.array([e_upper]),
                                 *args ),nround
                                   )
                         )[0]
    
    while integrand_upper != 0:
        e_upper += 0.1
        integrand_upper  = abs( np.round( f(np.array([e_upper]),
                                 *args ),nround
                                   )
                         )[0]
    
    
    
    
    return e_lower,e_upper




def current_elastic_voltage(Vlist,ef,
                      npoints_integrate,nround
                      ,H,dim,GammaL,GammaR,betaL,betaR,eta=1/2,plot_bool=False):
    
    Icurrent_array = np.zeros((len(Vlist),))
    for i in range(len(Vlist)):
        Vbias = Vlist[i]
        muL,muR = ef + Vbias*eta ,ef - Vbias*(1-eta)
        
        
        elist = [muL,muR]
        e_lower,e_upper = calc_bound_integrand_current(elist,nround,
                                   negf_method.integrand_current,
                                    H,dim,GammaL,GammaR,betaL,betaR,muL,muR)


        energies_integrate = np.linspace(e_lower,e_upper,npoints_integrate)


        current,integrand = trapz_integrate(energies_integrate,
                                   negf_method.integrand_current,
                                    H,dim,GammaL,GammaR,betaL,betaR,muL,muR)
        
        Icurrent_array[i] = current

        if plot_bool == True:
            plt.plot(energies_integrate,integrand)
            plt.show()
            
    return Icurrent_array


def current_inelastic_voltage(Vlist,ef,
                      npoints_integrate,nround
                      ,H,dim,GammaL,GammaR,betaL,betaR,lambda_vib_mat,hbaromega,eta=1/2,plot_bool=False):
    
    Icurrent_arr = np.zeros((len(Vlist),))
    for i in range(len(Vlist)):
        Vbias = Vlist[i]
        muL,muR = ef + Vbias*eta ,ef - Vbias*(1-eta)
        
        
        elist = [muL,muR,muL-hbaromega,muR-hbaromega]
        e_lower,e_upper = calc_bound_integrand_current(elist,nround,
                                   negf_method.integrand_current_inelastic,
                                                       H,dim,
                             GammaL,GammaR,
                              betaL,betaR,muL,muR,lambda_vib_mat,hbaromega)


        energies_integrate = np.linspace(e_lower,e_upper,npoints_integrate)


        current,integrand = trapz_integrate(energies_integrate,
                                   negf_method.integrand_current_inelastic,
                                            H,dim,GammaL,GammaR,
                                          betaL,betaR,muL,muR,
                                          lambda_vib_mat,hbaromega)
        
        Icurrent_arr[i] = current

        if plot_bool == True:
            plt.plot(energies_integrate,integrand)
            plt.show()
            
            
        
    return Icurrent_arr


def current_total(Vlist,ef,
                          npoints_integrate,nround
                          ,H,dim,GammaL,GammaR,betaL,betaR,lambda_vib_mat,hbaromega,eta=1/2,plot_bool=False,
                       nround_V0=12):

    ### elastic current: 
    Icurrent_elastic =  current_elastic_voltage(Vlist,ef,
                          npoints_integrate,nround
                          ,H,dim,GammaL,GammaR,betaL,betaR,eta,plot_bool)
    
    #### inelastic current:
    Icurrent_inelastic = current_inelastic_voltage(Vlist,ef,
                          npoints_integrate,nround
                          ,H,dim,GammaL,GammaR,betaL,betaR,lambda_vib_mat,hbaromega,eta,plot_bool)


    index_array = np.where(Vlist ==0.0)[0]

    IV0 = abs(np.round(Icurrent_inelastic[index_array[0]],nround_V0))

    assert IV0 == 0.0, 'Inelatic current is not zero at Vbias = 0: I(V=0) = {}'.format(IV0)
    
    
    
    return Icurrent_elastic,Icurrent_inelastic,Icurrent_elastic + Icurrent_inelastic



def differential_conductance_voltage(Vlist,ef,
                              npoints_integrate,nround
                              ,betaL,betaR,
                                Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega,eta,plot_bool=False):
    
    '''
    Differential conductance as a function of bias voltage
    
    ---------------- Parameters:
    
    Vlist - array of bias voltages
    npoints_integrate - number of energy points to integrate over
    nround - order to which integrand is neglected
    betaL,betaR = beta's of left, right lead.
    Hamiltonian0 = Hamiltonian of the scattering region, (dimXdim) array
    dim = integer
    GammaL,GammaR = Gamma coupling matrix to left,right lead
    lambda_vib_mat = vibrational coupling matrix 
    hbaromega = energy of vibrational modes
    eta = capactive coupling 
    plot_bool= if true plotes the integrand of the differential conductance.
    
    Output:
    
    Differential conductances for bias voltages in Vlist
    '''
    
    G1_array = np.zeros((len(Vlist),))
    for i in range(len(Vlist)):
        Vbias = Vlist[i]
        muL,muR = ef + Vbias*eta ,ef - Vbias*(1-eta)
        
        
        elist = [muL,muR,muL-hbaromega,muR-hbaromega]
        e_lower,e_upper = calc_bound_integrand_current(elist,nround,
                                   negf_method.linear_conductance_integrand,
                                    ef,eta,Vbias,betaL,betaR,
                                 Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega)


        energies_integrate = np.linspace(e_lower,e_upper,npoints_integrate)

        G1V,integrand = trapz_integrate(energies_integrate,
                                                           negf_method.linear_conductance_integrand,
                                    ef,eta,Vbias,betaL,betaR,
                                 Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega)
        
        G1_array[i] = G1V

        if plot_bool == True:
            plt.plot(energies_integrate,integrand)
            plt.show()
            
    return G1_array



def second_differential_conductance_voltage(Vlist,ef,
                              npoints_integrate,nround
                              ,betaL,betaR,
                                Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega,eta,plot_bool=False):
    
    '''
    Differential conductance as a function of bias voltage
    
    ---------------- Parameters:
    
    Vlist - array of bias voltages
    npoints_integrate - number of energy points to integrate over
    nround - order to which integrand is neglected
    betaL,betaR = beta's of left, right lead.
    Hamiltonian0 = Hamiltonian of the scattering region, (dimXdim) array
    dim = integer
    GammaL,GammaR = Gamma coupling matrix to left,right lead
    lambda_vib_mat = vibrational coupling matrix 
    hbaromega = energy of vibrational modes
    eta = capactive coupling 
    plot_bool= if true plotes the integrand of the differential conductance.
    
    Output:
    
    Differential conductances for bias voltages in Vlist
    '''
    
    G2_array = np.zeros((len(Vlist),))
    for i in range(len(Vlist)):
        Vbias = Vlist[i]
        muL,muR = ef + Vbias*eta ,ef - Vbias*(1-eta)
        
        
        elist = [muL,muR,muL-hbaromega,muR-hbaromega]
        e_lower,e_upper = calc_bound_integrand_current(elist,nround,
                                   negf_method.secondorder_conductance_integrand,
                                    ef,eta,Vbias,betaL,betaR,
                                 Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega)


        energies_integrate = np.linspace(e_lower,e_upper,npoints_integrate)

        G2V,integrand = trapz_integrate(energies_integrate,
                                                           negf_method.secondorder_conductance_integrand,
                                    ef,eta,Vbias,betaL,betaR,
                                 Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega)
        
        G2_array[i] = G2V

        if plot_bool == True:
            plt.plot(energies_integrate,integrand)
            plt.show()
            
    return G2_array



def func_bias_voltages(Vmax,Vnpoints):
    Vlist = np.linspace(-Vmax,Vmax,Vnpoints)
    boolzero = 0.0 in Vlist
    
        
    if boolzero == False:
        
        'Vbias = 0 must be in Bias voltages'
        Vlist = np.linspace(-Vmax,Vmax,Vnpoints+1)
    
        return Vlist
    
    if boolzero == True:
    
        return Vlist
    
    
def func_PC_list(y1list,y2list,xlist):
    '''Input
    y1list,y2list: lists that are a function of the parameter x in xlist
    Output
    plist = list with values: 'P = (y2-y1)/(y1 + y2)' 
    xprime_list = New x parameters. Values of x for which y1(x) + y2(x) =0 are removed (0/0 is numerically impossible)'''
    
    p_list = []
    xprime_list= []
    for i in range(len(xlist)):
        x = xlist[i]
        
        y1 = y1list[i]
        y2 = y2list[i]
        
        
        
        if x!=0:
            p_list.append(100*np.subtract(y1,y2)/(y2 + y1))
            
            xprime_list.append(x)
            
    
    return xprime_list,p_list