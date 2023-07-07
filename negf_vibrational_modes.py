#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman


import numpy as np 
from matplotlib import pyplot as plt


############################################################################################
############################################################################################
####################### Fermi - Dirac Function distrubtions & Derivatives ##################
############################################################################################
############################################################################################

def func_beta(T):
    kB = 8.617343*(10**(-5)) # eV/Kelvin
    ''' Input:
    -Temperature in Kelvin
    Output:
    - beta in eV^-1
    '''
    if T > 0:
        return 1/(kB*T)
    if T ==0:
        return np.infty
    if T<0:
        print('Temperature cannot be negative')

# Fermi-Dirac function
def fermi_dirac(energies,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The fermi-Dirac distribution for energy 
    '''
    
    return  1/(np.exp(beta*(energies-mui) ) + 1 )




#########################################################
############# Green's function
#########################################################


def GRA(energies, 
        H,dim,
        GammaL,
        GammaR
         ):
    
    '''
    Returns retarded & advanced Green's functions for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    
    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
       

    npoints_energy = len(energies)

    ### convert input matrices to tensors
    array_ones_energies = np.ones((npoints_energy))
    identity_matrix = np.identity(dim,dtype =complex)

    Energy_tensor = np.tensordot(energies,identity_matrix,axes =0)
    H_tensor     = np.tensordot(array_ones_energies,H,axes =0)
    Gamma_tensor = np.tensordot(array_ones_energies,GammaL + GammaR,axes =0)    
    
    ## Calculate Green's functions
    total = Energy_tensor - H_tensor + (1j/2)*Gamma_tensor 
    GR    = np.linalg.inv(total)                            ## retarded Green's function
    GA    = np.transpose(np.conjugate(GR),axes = (0,2,1))   ## advanced Green's function
    
    return GR,GA
    

#########################################################
############# Elastic Transmissions & currents
#########################################################
    
# Transmission left to right
# Only valid for 2terminal junctions
def TLR(energies, 
        H,dim,
        GammaL,
        GammaR):
    
    '''
    Returns transmission from left to right for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    
    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
    
    GR,GA = GRA(energies,H,dim,GammaL,GammaR)
    
    npoints_energy = len(energies)
    array_ones_energies = np.ones((npoints_energy))
    GammaL_tensor = np.tensordot(array_ones_energies,GammaL ,axes =0)
    GammaR_tensor = np.tensordot(array_ones_energies, GammaR,axes =0)


    T = np.matmul(np.matmul(np.matmul(GammaL_tensor,GA),GammaR_tensor),GR)
    TLRe = np.matrix.trace(T,axis1 = 1,axis2=2).real


    return TLRe



# Transmission left to right
# Only valid for 2terminal junctions
def TRL(energies, 
        H,dim,
        GammaL,
        GammaR):
    
    '''
    Returns transmission from right to left for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    
    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
    GR,GA = GRA(energies,H,dim,GammaL,GammaR)
    
    npoints_energy = len(energies)
    array_ones_energies = np.ones((npoints_energy))
    GammaL_tensor = np.tensordot(array_ones_energies,GammaL ,axes =0)
    GammaR_tensor = np.tensordot(array_ones_energies, GammaR,axes =0)

    T = np.matmul(np.matmul(np.matmul(GammaR_tensor,GA),GammaL_tensor),GR)
    TRLe = np.matrix.trace(T,axis1 = 1,axis2=2).real


    return TRLe



def integrand_current(energies, H,dim,
                             GammaL,GammaR,
                          betaL,betaR,muL,muR):
    
    '''
    Input:
    - energies = array of energies of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead

    Output:
    - Current calculated with Landauer-Buttiker Formula '''
        
    
    fL = fermi_dirac(energies,muL,betaL)
    fR = fermi_dirac(energies,muR,betaR)
    
    
    TLRe = TLR(energies, H,dim,GammaL,GammaR)
    TRLe = TRL(energies,H,dim,GammaL,GammaR)

    
    integrand = TLRe*fL-TRLe*fR 
    
    return integrand






#########################################################
############# Inelastic Green's function
#########################################################


def GRA_inel(energies, 
        H,dim,
        GammaL,
        GammaR,
        lambda_vib_mat
         ):
    
    '''
    Returns retarded & advanced Green's functions for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN  array)
    lambda_vib_mat = vibrational coupling matrix (NXN array)
    
    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
    npoints_energy = len(energies)

    ### convert input matrices to tensors
    array_ones_energies = np.ones((npoints_energy))
    identity_matrix = np.identity(dim,dtype =complex)

    Energy_tensor = np.tensordot(energies,identity_matrix,axes =0)
    H_tensor     = np.tensordot(array_ones_energies,H,axes =0)
    Gamma_tensor = np.tensordot(array_ones_energies,GammaL + GammaR,axes =0)    
    
    ## Calculate Green's functions
    total = Energy_tensor - H_tensor + (1j/2)*Gamma_tensor 
    GR    = np.linalg.inv(total)                            ## retarded inelastic Green's function
    
    
    #### lambda matrix tensor
    lambda_vib_tensor = np.tensordot(array_ones_energies,lambda_vib_mat ,axes =0)    
    
    GRinel   = np.matmul( GR,np.matmul(lambda_vib_tensor,GR))
    GAinel   = np.transpose(np.conjugate(GRinel),axes = (0,2,1))   ## advanced inelastic Green's function

    return GRinel,GAinel
    

#########################################################
############# Inelastic Transmissions & currents
#########################################################
    
# Transmission left to right
# Only valid for 2terminal junctions
def TLR_inel_energy(energies, 
        H,dim,
        GammaL,
        GammaR,
        lambda_vib_mat,hbaromega):
    
    '''
    Returns transmission from left to right for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    lambda_vib_matlambda_vib = vibrational coupling matrix (NXN array)

    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
    
    GRinel,GAinel = GRA_inel(energies,H,dim,GammaL,GammaR,lambda_vib_mat)
    
    npoints_energy = len(energies)
    array_ones_energies = np.ones((npoints_energy))
    GammaL_tensor = np.tensordot(array_ones_energies,GammaL ,axes =0)
    GammaR_tensor = np.tensordot(array_ones_energies, GammaR,axes =0)


    T_inel = np.matmul(np.matmul(np.matmul(GammaL_tensor,GAinel),GammaR_tensor),GRinel)
    Ttraced = np.matrix.trace(T_inel,axis1 = 2,axis2=1).real


    return Ttraced



# Transmission left to right
# Only valid for 2terminal junctions
def TRL_inel_energy(energies, 
        H,dim,
        GammaL,
        GammaR,lambda_vib_mat,hbaromega):
    
    '''
    Returns transmission from right to left for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    lambda_vib_matlambda_vib = vibrational coupling matrix (NXN array) of mode i
    hbaromega  = energy of vibrational mode i

    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
    GRinel,GAinel = GRA_inel(energies,H,dim,GammaL,GammaR,lambda_vib_mat)
    
    
    
    ###### Beware : For energy dependent leads, GammaL or GammaR depend on \hbar\omega as well.
    npoints_energy      = len(energies)
    array_ones_energies = np.ones((npoints_energy))
    GammaL_tensor       = np.tensordot(array_ones_energies,GammaL ,axes =0)
    GammaR_tensor       = np.tensordot(array_ones_energies, GammaR,axes =0)

    T_inel = np.matmul(np.matmul(np.matmul(GammaR_tensor,GAinel),GammaL_tensor),GRinel)
    Ttraced = np.matrix.trace(T_inel,axis1 = 1,axis2=2).real


    return Ttraced



def integrand_current_inelastic(energies, H,dim,
                             GammaL,GammaR,
                          betaL,betaR,muL,muR,lambda_vib_mat,hbaromega):
    
    '''
    Input:
    - energies = array of energies of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead

    Output:
    - Current calculated with Landauer-Buttiker Formula '''
        
    
    assert hbaromega >=0, '$\hbar \omega $ must be larger than or equal to zero, if not then the system is spontaneously vibrating.'
    
    fL = fermi_dirac(energies,muL,betaL)
    fR = fermi_dirac(energies,muR,betaR)
    
    fLp = fermi_dirac(energies-hbaromega,muL,betaL)
    fRp = fermi_dirac(energies-hbaromega,muR,betaR)
    
    
    TLRe = TLR_inel_energy(energies,H,dim,GammaL,GammaR,lambda_vib_mat,hbaromega)
    TRLe = TRL_inel_energy(energies,H,dim,GammaL,GammaR,lambda_vib_mat,hbaromega)

    
    integrand = TLRe*fL*(1-fRp)-TRLe*fR*(1-fLp) 
    
    return integrand


#########################################################
############# Linear conductance
#########################################################

# Derivative of Fermi-Dirac function
def fermi_prime_dirac(energies,mui,beta,mui_prime):
    '''
    First order derivative of Fermi-Dirac function with respect to bias voltage.

    
    Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    - mui_prime = derivative of chemical potential with respect to bias voltage

    Output:
    - The "derivative of the fermi-Dirac with respect to bias voltage" distribution for energy 
    '''
    fdp = mui_prime*(beta/4)*((1/np.cosh(beta*(energies-mui)/2))**2)
    return fdp


def fermi_pp_dirac(energy,mui,beta,mui_prime):
    
    
    '''
    Second order derivative of Fermi-Dirac function with respect to bias voltage.
    
    Input:
    - energy of the electron
    - mui = chemical potential
    - mui_prime = derivative of chemical potential with respect to bias voltage
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac with respect to bias voltage" distribution for energy 
    '''
    
    fdpp = (mui_prime**2)*(2*beta**2)*((np.sinh(0.5*beta*(energy-mui))/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui)))
    return fdpp


def linear_conductance_integrand(energies,ef,eta,Vbias,betaL,betaR,
                                Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega):
    
    
    ''' 
    The integrand for the linear conductance
    
    
    
    '''
    
    muL,muR = ef + eta*Vbias,ef + (eta-1)*Vbias
    
    
    fL = fermi_dirac(energies,muL,betaL)
    fR = fermi_dirac(energies,muR,betaR)

    fLp = fermi_dirac(energies-hbaromega,muL,betaL)
    fRp = fermi_dirac(energies-hbaromega,muR,betaR)


    dfLdV = fermi_prime_dirac(energies,muL,betaL, eta)
    dfRdV = fermi_prime_dirac(energies,muR,betaR, eta-1)

    dfLpdV = fermi_prime_dirac(energies-hbaromega,muL,betaL, eta)
    dfRpdV = fermi_prime_dirac(energies-hbaromega,muR,betaR, eta-1)
    

    
    TLR_el   = TLR(energies,Hamiltonian0,dim,GammaL,GammaR)
    TLR_inel = TLR_inel_energy(energies,Hamiltonian0,dim,GammaL,GammaR,lambda_vib_mat,hbaromega)
    TRL_inel = TRL_inel_energy(energies,Hamiltonian0,dim,GammaL,GammaR,lambda_vib_mat,hbaromega)


    G1_elastic   = TLR_el*(dfLdV - dfRdV)
    G1_inelastic = TLR_inel*( dfLdV*(1-fRp) - fL*dfRpdV ) - TRL_inel*(dfRdV*(1-fLp) - fR*dfLpdV) 
    
    return G1_elastic + G1_inelastic



def secondorder_conductance_integrand(energies,ef,eta,Vbias,betaL,betaR,
                                Hamiltonian0,dim,
                                 GammaL,GammaR,lambda_vib_mat,hbaromega):
    
    
    ''' 
    The integrand for the linear conductance
    
    
    
    '''
    
    muL,muR = ef + eta*Vbias,ef + (eta-1)*Vbias
    
    
    ##### Fermi -Dirac functions
    fL = fermi_dirac(energies,muL,betaL)
    fR = fermi_dirac(energies,muR,betaR)

    fLp = fermi_dirac(energies-hbaromega,muL,betaL)
    fRp = fermi_dirac(energies-hbaromega,muR,betaR)

    ##### First derivative of Fermi -Dirac functions
    dfLdV = fermi_prime_dirac(energies,muL,betaL, eta)
    dfRdV = fermi_prime_dirac(energies,muR,betaR, eta-1)
    
    dfLpdV = fermi_prime_dirac(energies-hbaromega,muL,betaL, eta)
    dfRpdV = fermi_prime_dirac(energies-hbaromega,muR,betaR, eta-1)
    
    
    ##### Second derivative Fermi -Dirac functions
    dfLdV2 = fermi_pp_dirac(energies,muL,betaL, eta)
    dfRdV2 = fermi_pp_dirac(energies,muR,betaR, eta-1)
    
    dfLpdV2 = fermi_pp_dirac(energies-hbaromega,muL,betaL, eta)
    dfRpdV2 = fermi_pp_dirac(energies-hbaromega,muR,betaR, eta-1)
    
    #### Transmission
    TLR_el   = TLR(energies,Hamiltonian0,dim,GammaL,GammaR)
    TLR_inel = TLR_inel_energy(energies,Hamiltonian0,dim,GammaL,GammaR,lambda_vib_mat,hbaromega)
    TRL_inel = TRL_inel_energy(energies,Hamiltonian0,dim,GammaL,GammaR,lambda_vib_mat,hbaromega)

    G2_elastic = TLR_el*(dfLdV2 - dfRdV2)
    G2_inelastic = TLR_inel*(dfLdV2*(1-fRp) - 2*dfLdV*dfRpdV - fL*dfRpdV2) \
                 - TRL_inel*(dfRdV2*(1-fLp) - 2*dfRdV*dfLpdV - fR*dfLpdV2) 
    
    return G2_elastic + G2_inelastic





