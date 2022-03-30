import numpy as np
import json
from taylor_approximation import taylor_approximate


class Emulator_Pells(object):
    def __init__(self, emufilename, order=4):
        self.order= order
        self.load(emufilename)
        self.cpars = np.zeros(4)
        self.p0tab, self.p2tab, self.p4tab = {}, {}, {}

        #
    def load(self, emufilename):
        '''Load the Taylor series from emufilename and repackage it into
           a dictionary of arrays.'''
        # Load Taylor series json file from emufilename:
        json_file = open(emufilename,'r')
        emu = json.load( json_file )
        json_file.close()
        
        # repackage into form we need, i.e. a dictionary of arrays
        self.emu_dict = {'kvec': np.array(emu['kvec']),\
                         'x0': emu['x0'],\
                         'derivs0': [np.array(ll) for ll in emu['derivs0']],\
                         'derivs2': [np.array(ll) for ll in emu['derivs2']],\
                         'derivs4': [np.array(ll) for ll in emu['derivs4']]}
        del(emu)


    def update_cosmo(self, cpars):
        '''If the cosmology is not the same as the old one, update the ptables.'''
        if not np.allclose(cpars, self.cpars):
            self.cpars = cpars
            
            self.p0tab = taylor_approximate(cpars,\
                                           self.emu_dict['x0'],\
                                           self.emu_dict['derivs0'], order=self.order)
            self.p2tab = taylor_approximate(cpars,\
                                           self.emu_dict['x0'],\
                                           self.emu_dict['derivs2'], order=self.order)
            self.p4tab = taylor_approximate(cpars,\
                                           self.emu_dict['x0'],\
                                           self.emu_dict['derivs4'], order=self.order)

    def combine_bias_terms_pkell(self, bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
            
            
    def __call__(self, cpars, bpars):
        '''Evaluate the Taylor series for the spectrum given by 'spectra'
           at the point given by 'params'.'''
        self.update_cosmo(cpars)
        
        
        pvec =  np.concatenate( ([1], bpars) )

        kvec = self.emu_dict['kvec']
        p0, p2, p4 = self.combine_bias_terms_pkell(bpars, self.p0tab, self.p2tab, self.p4tab)

        return kvec, p0, p2, p4
