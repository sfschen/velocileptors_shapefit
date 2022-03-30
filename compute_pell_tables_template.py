import numpy as np

from shapefit import shapefit_factor
from linear_theory import D_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )

sigma8_z0 = 0.82
ki, pi = np.loadtxt('/global/cscratch1/sd/sfschen/velocileptors_shapefit/Pk_Planck15_Table4.txt', unpack=True)

def compute_pell_tables(pars, klin=ki, plin=pi, zeff = 0.38, Omfid = 0.31, sigma80 = sigma8_z0, sigma8=None):

    '''
    Compute the velocileptors prediction for P_ell given initial power spectra ki, pi.
    
    Here sigma80 is the sigma8 of the linear power spectrum given, and sigma8 is the ``assumed'' one
    such that f(z) = fsigma8 / sigma8.
    
    If no sigma8 is given the compute sigma8 = sigma80 * D(z), where D(z) is computed in the fiducial cosmology.
    
    '''
    
    fsigma8, apar, aperp, m = pars
    
    # Compute the implied value of fz
    if sigma8 is None:
        sigma8 = sigma80 * D_of_a(1./(1+zeff),OmegaM=Omfid)
        
    fz = fsigma8 / sigma8
    
    plin = (sigma8/sigma80)**2 * pi * np.exp(shapefit_factor(ki, m))
    
    # Now do the RSD
    modPT = LPT_RSD(ki, plin, kIR=0.2,\
                cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
    modPT.make_pltable(fz, kv=kvec, apar=apar, aperp=aperp, ngauss=3)
    
    return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable
    