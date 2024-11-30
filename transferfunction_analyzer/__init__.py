"""
=========================
transferfunction_analyzer
=========================

Transferfucntion analyzer is an entity providing methods for analyzing 
transfer functions in time and frequency domain

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2024.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal as sig

import sympy as sp
sp.init_printing()

import plot_format 
plot_format.set_style('ieeetran')
import pdb

class transferfunction_analyzer(thesdk):
    @property
    def poles(self):
        if not hasattr(self,'_poles'):
            self._poles=[]
        return self._poles

    @poles.setter
    def poles(self,val):
        self._poles=val


    @property
    def zeros(self):
        if not hasattr(self,'_zeros'):
            self._zeros=[]
        return self._zeros

    @zeros.setter
    def zeros(self,val):
        self._zeros=val

    @property
    def gain(self):
        if not hasattr(self,'_gain'):
            self._gain=1
        return self._gain

    @gain.setter
    def gain(self,val):
        self._gain=val

    @property
    def time(self):
        if not hasattr(self,'_time'):
            self._time=np.linspace(0,1)
        return self._time


    #Transer function as
    @property
    def tfsym(self):
        nominator=self._g
        self.subsdict={self._g:self.gain}
        denominator=1
        
        for index in range(len(self.poles)):
           currpolesym=sp.symbols('p%s' %(index))
           if self.poles[index]==0:
               denominator*=self._s
           else:
               denominator*=(self._s/currpolesym+1)
           self.subsdict.update({currpolesym:self.poles[index]})
           

        for index in range(len(self.zeros)):
           currzerosym=sp.symbols('p%s' %(index))
           if self.zeros[index]==0:
               nominator*=self._s
           else:
               nominator*=(seflf._s/currzerosym+1)
           self.subsdict.update({currzerosym:self.zeros[index]})
        self._tfsym=nominator/denominator
        return self._tfsym

    @property
    def impsym(self):
        return sp.inverse_laplace_transform(self.tfsym,self._s,self._t)

    @property
    def stepsym(self):
        return sp.inverse_laplace_transform(1/self._s*self.tfsym,self._s,self._t)

    @property
    def tflatex(self):
        return sp.latex(self.tfsym)

    @property
    def implatex(self):
        return sp.latex(self.impsym)

    @property
    def steplatex(self):
        return sp.latex(self.stepsym)

    def imp(self,t):
       imp=sp.lambdify(self._t,self.stepsym.subs(self.subsdict))
       return np.array(imp(t))

    @property
    def impres(self):
       self._impulse=self.ht[1].reshape(1,-1)
       return self._stepres

       self.step=self.hts[1].reshape(1,-1)
    

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        self.model='py';

        self._t=sp.symbols('t', positive=True)
        self._s=sp.symbols('s')
        self._g=sp.symbols('g')

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        pass #Currently nohing to add



if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
    from  transferfunction_analyzer import *
    from  transferfunction_analyzer.controller import controller as transferfunction_analyzer_controller
    import pdb
    import math
    # Implement argument parser
    parser = argparse.ArgumentParser(description='Parse selectors')
    parser.add_argument('--show', dest='show', type=bool, nargs='?', const = True, 
            default=False,help='Show figures on screen')
    args=parser.parse_args()

