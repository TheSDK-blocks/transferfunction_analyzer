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
        """Time vector used for producing time domain responses

        Default: If pole exists, 100 samples from 0 to 10/p0
        """
        if not hasattr(self,'_time'):
            if len(self.poles) > 0:
                self._time=np.linspace(0,10/self.poles[0],num=100).rehape(-1,1)
            else:
                self._time=np.linspace(0,1,num=100).reshape(-1,1)
        return self._time

    @time.setter
    def time(self,val):
        self._time=np.array(val)

    @property
    def freq(self):
        """Frequency vector used for producing frequency responses

        Default: If pole exists, 1000 samples from 0.01p0 to 1000p0 (five decades) in log-scale.
        """
        if not hasattr(self,'_freq'):
            if len(self.poles) > 0:
                self._freq=np.logspace(0.01*self.poles[0],100*self.poles[0],num=100000)
            else:
                self._freq=np.logspace(0,1,num=100000)
        return self._freq
    @freq.setter
    def freq(self,val):
        self._freq=np.array(val).reshape(-1,1)

    @property
    def omega(self):
        """Angular frequency vector used for producing frequency responses

        Default: If pole exists, 1000 samples from 0.01p0 to 1000p0 (five decades) in log-scale.
        """
        return 2.0*np.pi*self._freq

    @property
    def subsdict(self):
        """Substitution dictionary for evaluating numerical values of system responses
        """
        self._subsdict={self._g:self.gain}
        for index in range(len(self.poles)):
           currpolesym=sp.symbols('p%s' %(index))
           self._subsdict.update({currpolesym:self.poles[index]})
        for index in range(len(self.zeros)):
           currzerosym=sp.symbols('z%s' %(index))
           self._subsdict.update({currzerosym:self.zeros[index]})
        return self._subsdict

    @property
    def tfsym(self):
        nominator=self._g
        denominator=1

        for index in range(len(self.poles)):
           currpolesym=sp.symbols('p%s' %(index))
           if self.poles[index]==0:
               denominator*=self._s
           else:
               denominator*=(self._s/currpolesym+1)

        for index in range(len(self.zeros)):
           currzerosym=sp.symbols('z%s' %(index))
           if self.zeros[index]==0:
               nominator*=self._s
           else:
               nominator*=(self._s/currzerosym+1)

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

    def imp(self,**kwargs):
        """Impulse response

        Parameters
        ----------
            t : [ Real ]

        Returns
        -------
            Impulse response vector.
        """
        t=kwargs.get('t',self.time)
        imp=sp.lambdify(self._t,self.impsym.subs(self.subsdict))
        return np.array(imp(t)).reshape(-1,1)

    def step(self,**kwargs):
        """Step response

        Parameters
        ----------
            t : [ Real ]

        Returns
        -------
            Step response vector.
        """
        t=kwargs.get('t',self.time)
        imp=sp.lambdify(self._t,self.stepsym.subs(self.subsdict))
        return np.array(imp(t)).reshape(-1,1)

    def tf(self,**kwargs):
        """Step response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Step response vector.
        """
        f=kwargs.get('f',self.freq)
        tf=sp.lambdify(self._s,self.tfsym.subs(self.subsdict))
        return np.array(tf(self.omega*1j)).reshape(-1,1)

    def tfabs(self,**kwargs):
        """Ampliitude response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Amplitude response vector.
        """
        f=kwargs.get('f',self.freq)
        return np.abs(self.tf(f=f)).reshape(-1,1)

    def tfphase(self,**kwargs):
        """Bode estimate of pole effects over the given frequency range.

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Phase response vector in Deg.
        """
        f=kwargs.get('f',self.freq)
        return np.angle(self.tf(f=f),deg=True).reshape(-1,1)

    def polesbodeamp(self,**kwargs):
        """Phase response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Bode estimate of poles efect to frequency response
        """
        f=kwargs.get('f',self.freq)
        omega=2*np.pi*f
        self._polesbodeamp=np.zeros((len(omega),len(self.poles)))
        for col in range(len(self.poles)):
            pole=self.poles[col]
            self._polesbodeamp[(omega >= pole)[:,0],col]=-10*np.log10(omega[omega >= pole]/np.abs(pole))
        self._polesbodeamp=10**(self._polesbodeamp/10)
        return self._polesbodeamp

    def polesbodephase(self,**kwargs):
        """Phase response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Bode estimate of poles efect to frequency response
        """
        f=kwargs.get('f',self.freq)
        omega=2*np.pi*f
        self._polesbodephase=np.zeros((len(omega),len(self.poles)))
        for col in range(len(self.poles)):
            pole=self.poles[col]
            self._polesbodephase[np.logical_and(omega >= 0.1*pole, omega <= 10*pole )[:,0],col]=-1*np.sign(pole)*45*np.log10(omega[np.logical_and(omega >= 0.1*pole,omega <= 10*pole)]/(0.1*pole))
            self._polesbodephase[(omega > 10*pole )[:,0],col]=-1*np.sign(pole)*90
        return self._polesbodephase

    def zerosbodeamp(self,**kwargs):
        """Phase response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Bode estimate of zeros efect to frequency response
        """
        f=kwargs.get('f',self.freq)
        omega=2*np.pi*f
        self._zerosbodeamp=np.zeros((len(omega),len(self.zeros)))
        for col in range(len(self.zeros)):
            zero=self.zeros[col]
            self._zerosbodeamp[(omega >= zero)[:,0],col]=10*np.log10(omega[omega >= zero]/np.abs(zero))
        self._zerosbodeamp=10**(self._zerosbodeamp/10)
        return self._zerosbodeamp

    def zerosbodephase(self,**kwargs):
        """Phase response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Bode estimate of zeros efect to frequency response
        """
        f=kwargs.get('f',self.freq)
        omega=2*np.pi*f
        self._zerosbodephase=np.zeros((len(omega),len(self.zeros)))
        for col in range(len(self.zeros)):
            zero=self.zeros[col]
            self._zerosbodephase[np.logical_and(omega >= 0.1*zero, omega <= 10*zero )[:,0],col]=1*np.sign(zero)*45*np.log10(omega[np.logical_and(omega >= 0.1*zero,omega <= 10*zero)]/(0.1*zero))
            self._zerosbodephase[(omega > 10*zero )[:,0],col]=1*np.sign(zero)*90
        return self._zerosbodephase

    def bodeamp(self,**kwargs):
        """Bode amplitude response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Bode estimate of amplitude response
        """
        f=kwargs.get('f',self.freq)
        self._bodeamp=self.gain*10**(np.sum(10*np.log10(self.polesbodeamp(f=f)),axis=1)/10+np.sum(10*np.log10(self.zerosbodeamp(f=f)),axis=1)/10).reshape(-1,1)
        return self._bodeamp

    def bodephase(self,**kwargs):
        """Phase response

        Parameters
        ----------
            f : [ Real ]

        Returns
        -------
            Bode estimate of phase response.
        """
        f=kwargs.get('f',self.freq)
        self._bodephase=(np.sum(self.polesbodephase(f=f),axis=1)+np.sum(self.zerosbodephase(f=f),axis=1)).reshape(-1,1)
        return self._bodephase

    def __init__(self,*arg):
        self.print_log(type='I', msg='Inititalizing %s' %(__name__))
        self.model='py';

        # These symbols we always have when handling system equations in Laplace domain
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
    import matplotlib.ticker as ticker
    from  transferfunction_analyzer import *
    import pdb
    import numpy as np
    # Implement argument parser
    parser = argparse.ArgumentParser(description='Parse selectors')
    parser.add_argument('--show', dest='show', type=bool, nargs='?', const = True,
            default=False,help='Show figures on screen')
    args=parser.parse_args()

    RC=1
    tfa=transferfunction_analyzer()
    tfa.poles=[RC]
    tfa.zeros=[10*RC]
    tfa.time=np.linspace(0,10*RC,num=100)
    tfa.freq=np.logspace(-3,3,base=10,num=100)/RC

    # Impulse response
    plt.figure()
    h=plt.subplot();
    plt.plot(tfa.time,tfa.imp(),label='RC=%s' %(RC));
    plt.ylim(0,1.1*np.max(tfa.imp()));
    h.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    h.xaxis.set_major_locator(ticker.MultipleLocator(base=RC))
    plt.xlim((0,tfa.time[-1]));
    plt.suptitle('Impulse response');
    plt.ylabel('h(t)');
    plt.xlabel(r'Relative time ( $\tau$ =RC)');
    lgd=plt.legend(loc='upper right');
    plt.grid(True);
    plt.savefig('./Single_pole_impulse_response.eps', format='eps', dpi=300);
    plt.show(block=False);

    #Step response
    plt.figure()
    h=plt.subplot();
    plt.plot(tfa.time,tfa.step(),label='RC=%s' %(RC));
    plt.ylim(0,1.1*np.max(tfa.step()));
    h.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    plt.xlim((0,tfa.time[-1]));
    h.xaxis.set_major_locator(ticker.MultipleLocator(base=RC))
    plt.suptitle('Step response');
    plt.ylabel('h(t)');
    plt.xlabel(r'Relative time ( $\tau$ =RC)');
    lgd=plt.legend(loc='upper right');
    plt.grid(True);
    plt.savefig('./Single_pole_step_response.eps', format='eps', dpi=300);
    plt.show(block=False);

    #Amplitude response
    plt.figure()
    h=plt.subplot();
    plt.semilogx(tfa.omega,tfa.tfabs(),label='RC=%s' %(RC));
    plt.ylim(0,1.1*np.max(tfa.tfabs()));
    h.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    # How to make xlim work with logscale plots
    #plt.xlim((0,tfa.omega[-1]));
    plt.suptitle('Amplitude response');
    plt.ylabel(r'$\left|H(f)\right|$');
    plt.xlabel(r'Relative frequency ( $p_0 =\frac{1}{RC}$)');
    lgd=plt.legend(loc='upper right');
    plt.grid(True);
    plt.savefig('./Single_pole_amplitude_response.eps', format='eps', dpi=300);
    plt.show(block=False);

    #Amplitude response in decibels
    plt.figure()
    h=plt.subplot();
    plt.ylim(-40,3);
    h.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
    # How to make xlim work with logscale plots
    #plt.xlim((0,tfa.omega[-1]));
    #plt.plot(tfa.freq,tfa.tfabs(),label='RC=%s' %(RC));
    plt.semilogx(tfa.omega,10*np.log10(tfa.tfabs()),label='RC=%s' %(RC));
    plt.suptitle('Amplitude response');
    plt.ylabel(r'$\left| H(f) \right|$ [dB]');
    plt.xlabel(r'Relative frequency ( $p_0 =\frac{1}{RC}$)');
    lgd=plt.legend(loc='upper right');
    plt.grid(True);
    plt.savefig('./Single_pole_amplitude_response_ind_db.eps', format='eps', dpi=300);
    plt.show(block=False);

    #Phase response in deg
    plt.figure()
    h=plt.subplot();
    plt.semilogx(tfa.omega,tfa.tfphase(),label='RC=%s' %(RC));
    plt.ylim(np.min(tfa.tfphase())-10,np.max(tfa.tfphase())+10);
    h.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
    # How to make xlim work with logscale plots
    #plt.xlim((0,tfa.omega[-1]));
    plt.suptitle('Phase response');
    plt.ylabel(r'$\angle$ H(f) [dB]');
    plt.xlabel(r'Relative frequency ( $p_0 =\frac{1}{RC}$)');
    lgd=plt.legend(loc='upper right');
    plt.grid(True);
    plt.savefig('./Single_phase_response.eps', format='eps', dpi=300);
    plt.show(block=False);

    #Bode plot of amplitude response in decibels
    plt.figure()
    h=plt.subplot();
    plt.ylim(-40,3);
    h.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
    # How to make xlim work with logscale plots
    #plt.xlim((0,tfa.omega[-1]));
    #plt.plot(tfa.freq,tfa.tfabs(),label='RC=%s' %(RC));
    plt.semilogx(tfa.omega,10*np.log10(tfa.bodeamp()),label='RC=%s' %(RC));
    plt.suptitle('Amplitude response');
    plt.ylabel(r'$\left| H(f) \right|$ [dB]');
    plt.xlabel(r'Relative frequency ( $p_0 =\frac{1}{RC}$)');
    lgd=plt.legend(loc='upper right');
    plt.grid(True);
    plt.savefig('./Single_pole_amplitude_response_ind_db.eps', format='eps', dpi=300);
    plt.show(block=False);

    #Bode plot of phase respnse
    plt.figure()
    h=plt.subplot();
    plt.semilogx(tfa.omega,tfa.bodephase(),label='RC=%s' %(RC));
    plt.ylim(np.min(tfa.bodephase())-10,np.max(tfa.bodephase())+10);
    h.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
    # How to make xlim work with logscale plots
    #plt.xlim((0,tfa.omega[-1]));
    plt.suptitle('Phase response');
    plt.ylabel(r'$\angle$ H(f) [dB]');
    plt.xlabel(r'Relative frequency ( $p_0 =\frac{1}{RC}$)');
    lgd=plt.legend(loc='upper right');
    plt.grid(True);
    plt.savefig('./Single_phase_response.eps', format='eps', dpi=300);
    plt.show(block=False);

    if args.show:
        input()


