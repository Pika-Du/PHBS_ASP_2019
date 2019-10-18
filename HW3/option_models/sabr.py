    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm

'''
Asymptotic approximation for 0<beta<=1 by Hagan
'''
def bsm_vol(strike, forward, texp, sigma, alpha=0, rho=0, beta=1):
    if(texp<=0.0):
        return( 0.0 )

    powFwdStrk = (forward*strike)**((1-beta)/2)
    logFwdStrk = np.log(forward/strike)
    logFwdStrk2 = logFwdStrk**2
    rho2 = rho*rho

    pre1 = powFwdStrk*( 1 + (1-beta)**2/24 * logFwdStrk2*(1 + (1-beta)**2/80 * logFwdStrk2) )
  
    pre2alp0 = (2-3*rho2)*alpha**2/24
    pre2alp1 = alpha*rho*beta/4/powFwdStrk
    pre2alp2 = (1-beta)**2/24/powFwdStrk**2

    pre2 = 1 + texp*( pre2alp0 + sigma*(pre2alp1 + pre2alp2*sigma) )

    zz = powFwdStrk*logFwdStrk*alpha/np.fmax(sigma, 1e-32)  # need to make sure sig > 0
    if isinstance(zz, float):
        zz = np.array([zz])
    yy = np.sqrt(1 + zz*(zz-2*rho))

    xx_zz = np.zeros(zz.size)

    ind = np.where(abs(zz) < 1e-5)
    xx_zz[ind] = 1 + (rho/2)*zz[ind] + (1/2*rho2-1/6)*zz[ind]**2 + 1/8*(5*rho2-3)*rho*zz[ind]**3
    ind = np.where(zz >= 1e-5)
    xx_zz[ind] = np.log( (yy[[ind]] + (zz[ind]-rho))/(1-rho) ) / zz[ind]
    ind = np.where(zz <= -1e-5)
    xx_zz[ind] = np.log( (1+rho)/(yy[ind] - (zz[ind]-rho)) ) / zz[ind]

    bsmvol = sigma*pre2/(pre1*xx_zz) # bsm vol
    return(bsmvol[0] if bsmvol.size==1 else bsmvol)

'''
Asymptotic approximation for beta=0 by Hagan
'''
def norm_vol(strike, forward, texp, sigma, alpha=0, rho=0):
    # forward, spot, sigma may be either scalar or np.array. 
    # texp, alpha, rho, beta should be scholar values

    if(texp<=0.0):
        return( 0.0 )
    
    zeta = (forward - strike)*alpha/np.fmax(sigma, 1e-32)
    # explicitly make np.array even if args are all scalar or list
    if isinstance(zeta, float):
        zeta = np.array([zeta])
        
    yy = np.sqrt(1 + zeta*(zeta - 2*rho))
    chi_zeta = np.zeros(zeta.size)
    
    rho2 = rho*rho
    ind = np.where(abs(zeta) < 1e-5)
    chi_zeta[ind] = 1 + 0.5*rho*zeta[ind] + (0.5*rho2 - 1/6)*zeta[ind]**2 + 1/8*(5*rho2-3)*rho*zeta[ind]**3

    ind = np.where(zeta >= 1e-5)
    chi_zeta[ind] = np.log( (yy[ind] + (zeta[ind] - rho))/(1-rho) ) / zeta[ind]

    ind = np.where(zeta <= -1e-5)
    chi_zeta[ind] = np.log( (1+rho)/(yy[ind] - (zeta[ind] - rho)) ) / zeta[ind]

    nvol = sigma * (1 + (2-3*rho2)/24*alpha**2*texp) / chi_zeta
 
    return(nvol[0] if nvol.size==1 else nvol)

'''
Hagan model class for 0<beta<=1
'''
class ModelHagan:
    alpha, beta, rho = 0.0, 1.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.beta = beta
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return bsm_vol(strike, forward, texp, sigma, alpha=self.alpha, beta=self.beta, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        bsm_vol = self.bsm_vol(strike, spot, texp, sigma)
        return self.bsm_model.price(strike, spot, texp, bsm_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        return sigma
    
    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or bsm vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp= self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        implvol=np.zeros(3)
        
        if(is_vol):
            implvol=price_or_vol3
        else:
            for i in range(3):
                implvol[i]=self.bsm_model.impvol(price_or_vol3[i],strike3[i],spot,texp,cp_sign = cp_sign)
        
        fun = lambda _calprmt:\
            bsm_vol(strike3, forward, texp, _calprmt[0], alpha = _calprmt[1], rho = _calprmt[2])-implvol
        para = sopt.root(fun, [0.1,0.1,0]).x
        return para # sigma, alpha, rho

'''
Hagan model class for beta=0
'''
class ModelNormalHagan:
    alpha, beta, rho = 0.0, 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.beta = 0.0 # not used but put it here
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return norm_vol(strike, forward, texp, sigma, alpha=self.alpha, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        n_vol = self.norm_vol(strike, spot, texp, sigma)
        return self.normal_model.price(strike, spot, texp, n_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            norm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 50)
        if(setval):
            self.sigma = sigma
        return sigma

    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or normal vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp= self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        implvol=np.zeros(3)
        
        if(is_vol):
            implvol=price_or_vol3
        else:
            for i in range(3):
                implvol[i]=self.normal_model.impvol(price_or_vol3[i],strike3[i],spot,texp,cp_sign = cp_sign)
        
        fun = lambda _calprmt:\
            norm_vol(strike3, forward, texp, _calprmt[0], alpha = _calprmt[1], rho = _calprmt[2])-implvol
        para = sopt.root(fun, [0.1*forward,0.1,0]).x
        return para # sigma, alpha, rho


'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    sample = 10000
    tstep = 100
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0, sample = 10000, tstep = 100):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.sample = sample
        self.tstep=tstep
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp, sigma, cp_sign=cp_sign)
        bsmmc_vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        
        return bsmmc_vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac *div_fac
        tstep =self.tstep
        d_t = texp/tstep
        
        np.random.seed(12345)
        
        volmc = np.ones((self.sample, tstep+1))
        Z1 = np.random.normal(size=(self.sample, tstep))
        volmc[:, 1:] = np.cumsum(self.alpha*np.sqrt(d_t)*Z1 - 0.5*self.alpha**2*d_t,axis =1)
        volmc = sigma*np.exp(volmc[:,: -1])
        
        np.random.seed(54321)
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(size = (self.sample, tstep))
        pricemc = np.cumsum(volmc*np.sqrt(d_t)*Z2 - 0.5*volmc**2*d_t, axis = 1)
        price = np.mean( np.fmax( cp_sign * (forward * np.exp( pricemc[:,-1:][:,None]) - strike), 0),axis=0)
                           
        
        return price*disc_fac

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    sample = 10000
    tstep = 100
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0,sample = 10000, tstep = 100):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.sample = sample
        self.tstep=tstep
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp, sigma, cp_sign=cp_sign)
        normmc_vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        
        return normmc_vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac *div_fac
        tstep =self.tstep
        d_t = texp/tstep
        
        np.random.seed(12345)
        
        volmc = np.ones((self.sample, tstep+1))
        Z1 = np.random.normal(size=(self.sample, tstep))
        volmc[:, 1:] = np.cumsum(self.alpha*np.sqrt(d_t)*Z1 - 0.5*self.alpha**2*d_t,axis =1)
        volmc = sigma*np.exp(volmc[:, :-1])
        
        np.random.seed(54321)
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(size = (self.sample, tstep))
        pricemc = forward+ np.cumsum(volmc*np.sqrt(d_t)*Z2, axis = 1)
        
        price = np.mean( np.fmax( cp_sign * (pricemc[:,-1][:,None] - strike), 0),axis=0 )
                           
        return price*disc_fac


'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    sample = 10000
    tstep = 100
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0, sample=10000, tstep=100):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.sample = sample
        self.tstep= tstep
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste) 
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp, sigma, cp_sign=cp_sign)
        bsmmc_vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        
        return bsmmc_vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        tstep =self.tstep
        d_t = texp/tstep
        
        np.random.seed(12345)
        volmc = np.ones((self.sample, tstep+1))
        Z1 = np.random.normal(size=(self.sample, tstep))

        volmc[:,1:] = np.cumprod(np.exp(self.alpha*np.sqrt(d_t)*Z1 - 0.5*self.alpha**2*d_t), axis = 1)
        volmc = sigma * volmc
        
        
        sweight = np.ones(tstep + 1)
        sweight[1::2] = 4
        sweight[2::2] = 2
        sweight[-1] = 1
        sweight = np.resize(sweight, (self.sample, tstep + 1))
        
        IV= d_t/3 * np.cumsum(sweight * volmc**2, axis =1)
        
        ConMC_bsmspot = spot * np.exp(self.rho/self.alpha*(volmc[:,-1]-volmc[:,0])-self.rho**2/2*IV[:,-1])
        ConMC_bsmvol = np.sqrt((1-self.rho**2)*IV[:,-1]/texp)
        ConMC_bsmprice = self.bsm_model.price(strike, ConMC_bsmspot[:,None], texp, ConMC_bsmvol[:,None], cp_sign=cp_sign)
        price = np.mean(ConMC_bsmprice, axis = 0)
        return price


'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    sample = 10000
    tstep = 100
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0, sample = 10000, tstep = 100):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.sample = sample
        self.tstep= tsteo
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None,cp_sign=1):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp, sigma, cp_sign=cp_sign)
        normmc_vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        
        return normmc_vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        tstep =self.tstep
        d_t = texp/tstep
        
        np.random.seed(12345)
        volmc2 = np.ones((self.sample, tstep+1))
        Z1 = np.random.normal(size=(self.sample, tstep))

        volmc2[:,1:] = np.cumprod(np.exp(self.alpha*np.sqrt(d_t)*Z1 - 0.5*self.alpha**2*d_t), axis = 1)
        volmc2 = sigma * volmc2
        
        sweight = np.ones(tstep + 1)
        sweight[1:-1] = 2
        sweight = np.resize(sweight, (self.sample, tstep + 1))
        
        IV= d_t/2 * np.cumsum(sweight * volmc2**2, axis =1)
        
        ConMC_normspot = spot + self.rho/self.alpha*(volmc2[:,-1]-volmc2[:,0])
        ConMC_normvol = np.sqrt((1-self.rho**2)*IV[:,-1]/texp)
        ConMC_normprice = self.normal_model.price(strike, ConMC_normspot[:,None], texp, ConMC_normvol[:,None], cp_sign=cp_sign)
        price = np.mean(ConMC_normprice, axis = 0)
        return price
