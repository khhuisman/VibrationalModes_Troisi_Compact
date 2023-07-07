
#Author: Karssien Hero Huisman

import numpy as np
import matplotlib.pyplot as plt

import kwant
import kwant.qsymm
import tinyarray

# define Pauli-matrices for convenience
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
a=1

# Import Geometrical Paramters for construction of SOC Hamiltonian.
import Geometry_Git


def make_system_straight_U0(Lm,epsilon,t):
    '''
    Input = Lenght of molecule
    epsilon = onsite energy
    t = NN hopping paramter (spin-independent)
    Output:
    - Kwant system of chain with spin depedent hopping
    '''
    
    a   = 1
    lat =  kwant.lattice.square(a,norbs = 2) # lattice with 2 spin degree of freedom
    syst = kwant.Builder()
    
    
    
     ### DEFINE LATTICE HAMILTONIAN ###
    for i in range( 1, Lm +1 ):
            syst[lat(i, 0)] =  epsilon*sigma_0 
            
           
              
           
    # hopping in z-direction
    syst[kwant.builder.HoppingKind((1,0), lat, lat)] =  -t*sigma_0 


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
        
        
        sigma_vec = [sigma_x,sigma_y,sigma_z]

      
        vvector = Geometry_Git.Vmvec(m=m1,stilde=1,a=a,c=c,M=M,N=N,chirality=chirality)
        innerproduct = 1j*lambda1*(vvector[0]*sigma_vec[0] + vvector[1]*sigma_vec[1] + vvector[2]*sigma_vec[2] )

        return innerproduct
    
    
    
    # NNN spin orbit coupling
    syst[kwant.builder.HoppingKind((2,0), lat, lat)] =  hopping_2site

    return syst

def make_system_no_soc(Lm,epsilon,t):
    '''
    Input = Lenght of molecule
    epsilon = onsite energy
    t = NN hopping paramter (spin-independent)
    Output:
    - Kwant system of chain with spin depedent hopping
    '''
    a   = 1
    lat =  kwant.lattice.square(a,norbs = 2) # lattice with 2 spin degree of freedom
    syst = kwant.Builder()
    
    
    
     ### DEFINE LATTICE HAMILTONIAN ###
    for i in range( 0, Lm  ):
            #Onsite Hubbard
            syst[lat(i, 0)] =  epsilon*sigma_0
      
    # hopping in z-direction
    syst[kwant.builder.HoppingKind((1,0), lat, lat)] =  -t*sigma_0 

    return syst

def vibmode_NNNsoc(Lm,epsilon,t, lambda1,chirality,a,c,M,N):
    
    '''Input
    system paramters
    Output:
    - vibrational matrix with NNN soc and normal NN hopping
    '''
    
    
         
    if lambda1 !=0.0:
        system =  make_system_straight_U0(Lm,epsilon,t)

        kwant_sytem = kwant.qsymm.builder_to_model(system, params={'lambda1': lambda1,
                                                                   'chirality':chirality,
                                                                   'a':a,'c':c, 'M':M,'N':N}) 


    if lambda1 ==0.0:
        system =  make_system_no_soc(Lm,epsilon,t)


        kwant_sytem = kwant.qsymm.builder_to_model(system) 


    Hamiltonian = np.array(kwant_sytem[1])
    hamiltonian_shape = Hamiltonian.shape
    mshape = hamiltonian_shape[0]

    return Hamiltonian


    
def make_system_straight_U0_NN(Lm,epsilon):
    '''
    Input = Lenght of molecule
    epsilon = onsite energy
    t = NN hopping paramter (spin-independent)
    Output:
    - Kwant system of chain with spin depedent hopping
    '''
    
    a   = 1
    lat =  kwant.lattice.square(a,norbs = 2) # lattice with 2 spin degree of freedom
    syst = kwant.Builder()
    
    
    
     ### DEFINE LATTICE HAMILTONIAN ###
    for i in range( 1, Lm +1 ):
            syst[lat(i, 0)] =  epsilon*sigma_0 
            
    def hopping_2site(site1, site2,t,lambda1,chirality,a,c,M,N):
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
        
        
        sigma_vec = [sigma_x,sigma_y,sigma_z]

      
        vvector = Geometry_Git.Vmvec(m=m1,stilde=1,a=a,c=c,M=M,N=N,chirality=chirality)
        innerproduct = 1j*lambda1*(vvector[0]*sigma_vec[0] + vvector[1]*sigma_vec[1] + vvector[2]*sigma_vec[2] )

        return innerproduct -t*sigma_0 
    
    
    
    # NNN spin orbit coupling
    syst[kwant.builder.HoppingKind((1,0), lat, lat)] =  hopping_2site

    return syst


def vibmode_NNsoc(Lm,epsilon,t, lambda1,chirality,a,c,M,N
                        ):
    
    '''Input
    system paramters
    Output:
    - vibrational matrix with NN soc and normal NN hopping
    '''
    
    
         
    
    system =  make_system_straight_U0_NN(Lm,epsilon)

    kwant_sytem = kwant.qsymm.builder_to_model(system, params={'t':t,'lambda1': lambda1,
                                                               'chirality':chirality,
                                                               'a':a,'c':c, 'M':M,'N':N}) 


   

    Hamiltonian = np.array(kwant_sytem[1])
    hamiltonian_shape = Hamiltonian.shape
    mshape = hamiltonian_shape[0]

    return Hamiltonian





