#!/usr/bin/env python
# coding: utf-8

# In[15]:


from matplotlib import pyplot as plt
import numpy as np
import kwant
import Sgeom_scatteringregion as Sgeom


def make_system_random(Lm ,Wg):
    
    '''
    Input:
    t    = NN hoppings
    tsom = NN hopping due to SOC
    Lm   = Length of S - shape
    Wg   = Width of S-shape
    Ouput
    kwant system of S-shape
    '''
    
    
    #Define hoppings in z,x direction
    # The z - direction lies along lead direction
    # The x - direction is defined in-plane orthogonal to the z-direction
    # The y - direction points out-of-plane
    # Thus the lattice has coordinate label (z,x,y) (instead of the usual (x,y,z))
    
    # Create Lattice
    
    a   = 1
    lat =  kwant.lattice.square(a,norbs = 1) # 2 d.o.f. per site => norbs = 2
    syst = kwant.Builder()
    
    epsilon0 = 0        #onsite energy
          #molecule length
    
     ### DEFINE LATTICE HAMILTONIAN ###
        
    
    d = Wg
    for i in range( -Lm , Lm + 1  ):
        for j in range(-d, d):
            
            
           
            # Sgeom is created:     
            if 0 <= i <= Lm and 0 <= j  <= Wg:
                syst[lat(i, j)] =  0
            if -Lm+1 <= i <= 1 and -Wg+1 <= j  <= 0:
                syst[lat(i, j)] = 0
              
           
    # hopping in z-direction
    syst[kwant.builder.HoppingKind((1,0), lat, lat)] =  1
    #hopping in x direction
    syst[kwant.builder.HoppingKind((0,1), lat, lat)] =  1
    
    def hopping_2site(site1, site2,lambda1,chirality,a,c,M,N):
        '''
        Input:
        - N = number of sites per lap
        - M = number of laps
        - c = length of the molecule
        - a = radius
        - m = site label
        - chirality = boolean indicating the chirality of the helix.
        Ouput:
        - Hopping matrices (in spin space) between site1,site2 for chirality = True
        '''
        m1,m2 = site1.pos
        j1,j2 = site2.pos
        
        value = np.random.rand(1)[0]
        
        return value
    
      # hopping in z-direction
    syst[kwant.builder.HoppingKind((1,0), lat, lat)] =  hopping_2site
    #hopping in x direction
    syst[kwant.builder.HoppingKind((0,1), lat, lat)] =  hopping_2site
    
    
    return syst


def lambda_matrix_random(L,Wg,plotbool=False):
    

    '''
    Returns Lambda hopping matrix with random entries. This matrix is time-reversal symmetric.
    
    
    ---------------------------
    Input:
    L,Wg = geometric paramters of S shape

    Ouput:
    Lambda    = Lambda hopping matrix with random entries. This matrix is time-reversal symmetric.
    
    '''
    
    hels3D =  make_system_random(L ,Wg) 
    
    
    if plotbool == True:
        kwant.plot(hels3D);
    
    Lambda_matrix_random = kwant.qsymm.builder_to_model(hels3D)[1]
    
    

    return np.kron(Lambda_matrix_random,np.identity(2))





def lambda_vib_constant(L,Wg, lambda_vib,lambda_vib_soc,plotbool=False):
    

    '''
    Returns Lambda hopping matrix with constant entries. This matrix is time-reversal symmetric.
    
    
    ---------------------------
    Input:
    L,Wg = geometric paramters of S shape
    lambda_vib = coupling parameter
    lambda_vib_soc = coupling parameter with soc
    

    Ouput:
    Lambda    = Lambda hopping matrix with constant entries. This matrix is time-reversal symmetric.
    
    '''
    
    hels3D =  Sgeom.make_system_U0(lambda_vib,lambda_vib_soc,L ,Wg) 
    
    
    if plotbool == True:
        kwant.plot(hels3D);
    
    lambda_mat_constant = kwant.qsymm.builder_to_model(hels3D)[1]
    
    

    return lambda_mat_constant




