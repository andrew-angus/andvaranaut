#!/bin/python3

import numpy as np
import scipy.stats as st
from time import time as stopwatch
import matplotlib.pyplot as plt
import os
import copy
from sklearn.model_selection import train_test_split
from functools import partial
from andvaranaut.core import _core,save_object
from andvaranaut.lhc import LHC
from andvaranaut.transform import wgp,kumaraswamy
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.optimize import Bounds, differential_evolution, minimize
from scipy.stats._continuous_distns import uniform_gen, norm_gen, truncnorm_gen
import re


# Identity conversions
class _none_conrev:
  def con(self,x):
   return x 
  def rev(self,x):
   return x 

# Inherit from surrogate class and add GP specific methods
class GPMCMC(LHC):
  def __init__(self,xconrevs=None,yconrevs=None,\
      kernel='RBF',noise=True,mean=0,**kwargs):
    # Call LHC init, then validate and set now data conversion/reversion attributes
    super().__init__(**kwargs)
    self.xc = copy.deepcopy(self.x)
    self.yc = copy.deepcopy(self.y)
    self.__conrev_check(xconrevs,yconrevs)
    self.change_model(kernel,noise,mean)
    self.__scrub_train_test()
    self.ym = copy.deepcopy(self.y)

  # Zero mean function
  def zero_mean(self,x):
    return np.zeros(self.ny)

  # Conversion of last n samples
  def __con(self,nsamps):
    self.xc = np.r_[self.xc,np.zeros((nsamps,self.nx))]
    self.yc = np.r_[self.yc,np.zeros((nsamps,self.ny))]
    for i in range(self.nx):
      self.xc[-nsamps:,i] = self.xconrevs[i].con(self.x[-nsamps:,i])
    for i in range(self.ny):
      self.yc[-nsamps:,i] = \
          self.yconrevs[i].con(self.y[-nsamps:,i]-self.ym[-nsamps:,i])

  # Inherit from lhc __del_samples and add converted dataset deletion
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    returned = super()._LHC__del_samples(ndels,method,idx,returns=True)
    if method == 'coarse_lhc':
      for i in range(ndels):
        self.xc = np.delete(self.xc,returned[i],axis=0)
        self.yc = np.delete(self.yc,returned[i],axis=0)
        self.ym = np.delete(self.ym,returned[i],axis=0)
    elif method == 'random':
      self.xc = self.xc[returned,:]
      self.yc = self.yc[returned,:]
      self.ym = self.ym[returned,:]
    elif method == 'specific':
      self.xc = self.xc[returned]
      self.yc = self.yc[returned]
      self.ym = self.ym[returned]
    self.__scrub_train_test()

  # Allow for changing conversion/reversion methods
  def change_conrevs(self,xconrevs=None,yconrevs=None):
    # Check and set new lists, then update converted datasets
    self.__conrev_check(xconrevs,yconrevs)
    for i in range(self.nx):
      self.xc[:,i] = self.xconrevs[i].con(self.x[:,i])
    for i in range(self.ny):
      self.yc[:,i] = self.yconrevs[i].con(self.y[:,i]-self.ym[:,i])

  # Allow for changing conversion/reversion methods
  def change_xconrevs(self,xconrevs=None):
    # Check and set new lists, then update converted datasets
    self.__conrev_check(xconrevs,yconrevs=self.yconrevs)
    for i in range(self.nx):
      self.xc[:,i] = self.xconrevs[i].con(self.x[:,i])

  # Allow for changing conversion/reversion methods
  def change_yconrevs(self,yconrevs=None):
    # Check and set new lists, then update converted datasets
    self.__conrev_check(self.xconrevs,yconrevs)
    for i in range(self.ny):
      self.yc[:,i] = self.yconrevs[i].con(self.y[:,i]-self.ym[:,i])

  # Converison/reversion input checking and setting (used in __init__ and change_conrevs)
  def __conrev_check(self,xconrevs,yconrevs):
    if xconrevs is None:
      xconrevs = [None for i in range(self.nx)]
    if yconrevs is None:
      yconrevs = [None for i in range(self.ny)]
    if not isinstance(xconrevs,list) or len(xconrevs) != self.nx:
      raise Exception(\
          "Error: xconrevs must be None or list of conversion/reversion classes of size nx")
    if not isinstance(yconrevs,list) or len(yconrevs) != self.ny:
      raise Exception(\
          "Error: xconrevs must be None or list of conversion/reversion classes of size nx")
    for j,i in enumerate(xconrevs+yconrevs):
      if (i is not None) and ((not callable(i.con)) or (not callable(i.rev))):
        raise Exception(\
            'Error: Provided data conversion/reversion function not callable.')
      elif i is None:
        if j < self.nx:
          xconrevs[j] = _none_conrev()
        else:
          yconrevs[j-self.nx] = _none_conrev()
    self.xconrevs = xconrevs
    self.yconrevs = yconrevs

  # Optionally set x and y attributes with existing datasets
  def set_data(self,x,y):
    super().set_data(x,y)
    self.xc = np.empty((0,self.nx))
    self.yc = np.empty((0,self.ny))

    # Evaluate mean function
    xm,ym = self._core__vector_solver(self.x,self.mean)
    if len(xm) != len(self.x):
      raise Exception("Mean function not valid at every x point in dataset")
    self.ym = np.r_[self.ym,ym]

    # Get converted values
    self.__con(self.nsamp)

    # Scrub train test
    self.__scrub_train_test()

  # Inherit and extend y_dist to have dist by surrogate predictions
  def y_dist(self,mode='hist_kde',nsamps=None,return_data=False,surrogate=True,**kwargs):
    # Allow for use of surrogate evaluations or underlying datasets
    if surrogate:
      xsamps = super()._LHC__latin_sample(nsamps)
      ypreds = self.predict(xsamps)
      super()._LHC__y_dist(ypreds,mode,**kwargs)
      if return_data:
        return xsamps,ypreds
    elif not surrogate:
      super().y_dist(mode,**kwargs)
    else:
      raise Exception("Error: surrogate argument must be of type bool")

  def __scrub_train_test(self):
    self.train = None
    self.test = None

  # Sample method inherits lhc sampling and adds adaptive sampling option
  def sample(self,nsamps,seed=None):
    # Evaluate samples
    super().sample(nsamps=nsamps,seed=seed)

    # Evaluate mean function
    xm,ym = self._core__vector_solver(self.x,self.mean)
    if len(xm) != len(self.x):
      raise Exception("Mean function not valid at every x point in dataset")
    self.ym = ym #TODO should not run solver for all x, only new samps

    # Get converted data
    self.xc = np.empty((0,self.nx))
    self.yc = np.empty((0,self.ny))
    self.nsamp = len(xm)
    self.__con(self.nsamp)

  # Fit GP standard method
  def fit(self,method='map',return_data=False,\
      iwgp=False,cwgp=False,jitter=1e-6,truncate=False,\
      restarts=1,**kwargs):
    self.m, self.gp, self.hypers, data = self.__fit(self.x,self.y-self.ym,\
        method,iwgp,cwgp,jitter,truncate,restarts,**kwargs)

    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,method,iwgp,cwgp,jitter=1e-6,truncate=False,restarts=1,\
      **kwargs):
    
    # PyMC context manager
    m = pm.Model()
    nsamp = len(y)
    with m:
      # Priors on GP hyperparameters
      if self.noise:
        if truncate:
          gvar = pm.Normal.dist(mu=0.0,sigma=1e-3)
          gvar = pm.Truncated('gv',gvar,lower=1e-15,upper=1.0)
        else:
          gvar = pm.HalfNormal('gv',sigma=1e-3)
      else:
        gvar = 0.0
      if truncate:
        kls = pm.TruncatedNormal('l',mu=0.5,sigma=0.15,\
            lower=1e-3,upper=100.0,shape=self.nx*self.nkern)
        kvar = pm.TruncatedNormal('kv',mu=1.0,sigma=0.15,\
            lower=1e-1,upper=100.0,shape=self.nkern)
      else:
        kls = pm.LogNormal('l',mu=0.0,sigma=1.0,shape=self.nx*self.nkern)
        kvar = pm.LogNormal('kv',mu=0.56,sigma=0.75,shape=self.nkern)

      # Input warping
      if iwgp:
        rc = 0
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            rc += self.xconrevs[i].np
        if truncate:
          iwgp = pm.TruncatedNormal('iwgp',mu=1.0,sigma=1.0,\
              lower=1e-3,upper=5.0,shape=rc)
        else:
          iwgp = pm.LogNormal('iwgp',mu=0.0,sigma=0.25,shape=rc)
        rc = 0
        x1 = []
        check = True
        warpers = self.iwgp_set(iwgp,mode='pytensor',x=x)
        for i in range(self.nx):
          if isinstance(warpers[i],wgp):
            check = False
            x1.append(warpers[i].conmc(x[:,i]))
          else:
            x1.append(self.xconrevs[i].con(x[:,i]))
          xin = pt.stack(x1,axis=1)
        if check:
          raise Exception('Error: iwgp set to true but none of xconrevs are wgp classes')
      else:
        xin = np.zeros_like(x)
        for i in range(self.nx):
          xin[:,i] = self.xconrevs[i].con(x[:,i])

      # Output warping
      if cwgp:
        if not isinstance(self.yconrevs[0],wgp):
          raise Exception('Error: cwgp set to true but yconrevs class is not wgp')
        npar = self.yconrevs[0].np
        if npar == 0:
          raise Exception('Error: cwgp set to true but wgp class has no tuneable parameters')
        rc = 0
        rcpos = 0
        for i in range(npar):
          if self.yconrevs[0].pos[i]:
            rcpos += 1
          else:
            rc += 1
        if rcpos > 0:
          if truncate:
            cwgpp = pm.TruncatedNormal('cwgp_pos',mu=1.0,sigma=1.0,\
                lower=1e-3,upper=5.0,shape=rcpos)
          else:
            cwgpp = pm.LogNormal('cwgp_pos',mu=0.0,sigma=0.25,shape=rcpos)
        if rc > 0:
          if truncate:
            cwgp = pm.TruncatedNormal('cwgp',mu=0.0,sigma=1.0,\
                lower=-10,upper=10,shape=rc)
          else:
            cwgp = pm.Normal('cwgp',mu=0.0,sigma=1.0,shape=rc)
        rc = 0
        rcpos = 0
        rvs = []
        for i in range(npar):
          if self.yconrevs[0].pos[i]:
            rvs.append(cwgpp[rcpos])
            rcpos += 1
          else:
            rvs.append(cwgp[rc])
            rc += 1
        warper = self.cwgp_set(rvs,mode='pytensor',y=y)
        yin = warper.conmc(y[:,0])
        yder = warper.dermc(y[:,0])
      else:
        yin = self.yconrevs[0].con(y[:,0])

      # Setup kernel, looping through all specified components
      for i in range(self.nkern):
        if self.kerns[i] == 'RBF':
          kerni = kvar[i]*pm.gp.cov.ExpQuad(self.nx,\
              ls=kls[i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'RatQuad':
          # Only works of only one ratquad kernel specified 
          alpha = pm.LogNormal('alpha',mu=0.56,sigma=0.75)
          kerni = kvar[i]*pm.gp.cov.RatQuad(self.nx,alpha=alpha,\
              ls=kls[i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'Matern52':
          kerni = kvar[i]*pm.gp.cov.Matern52(self.nx,\
              ls=kls[i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'Matern32':
          kerni = kvar[i]*pm.gp.cov.Matern32(self.nx,\
              ls=kls[i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'Exponential':
          kerni = kvar[i]*pm.gp.cov.Exponential(self.nx,\
              ls=kls[i*self.nx:(i+1)*self.nx])

        # Apply kernel operations if more than one kern specified
        if i == 0:
          kern = kerni
        elif self.ops[i-1] == '+':
          kern += kerni
        elif self.ops[i-1] == '*':
          kern *= kerni

      # GP and likelihood
      if cwgp:
        K = kern(xin)
        K += pt.identity_like(K)*(jitter+gvar)
        L = pt.slinalg.cholesky(K)
        beta = pt.slinalg.solve_triangular(L,yin,lower=True)
        alpha = pt.slinalg.solve_triangular(L.T,beta)
        y_ = pm.Potential('yopt',-0.5*pt.dot(yin.T,alpha)\
            -pt.sum(pt.log(pt.diag(L)))\
            -0.5*nsamp*pt.log(2*np.pi)\
            +pt.sum(pt.log(yder)))
      else:
        gp = pm.gp.Marginal(cov_func=kern)
        y_ = gp.marginal_likelihood("yopt", X=xin, y=yin, sigma=pt.sqrt(gvar), \
            jitter=jitter)

      # Fit and process results depending on method
      if method == 'map':
        maxl = np.finfo(np.float64).min
        if restarts > 1:
          for i in range(restarts):
            start = {str(ky):np.random.normal() for ky in m.cont_vars}
            try:
              data = pm.find_MAP(**kwargs)
              mp = copy.deepcopy(data)
              mpcheck = {str(ky):mp[str(ky)] for ky in m.cont_vars}
              logps = m.point_logps(point=mpcheck)
              logsum = np.sum(np.array([logps[str(ky)] for ky in logps.keys()]))
            except:
              print('Restart failed')
              logsum = maxl
            if logsum > maxl:
              mpmax = mp
              maxl = logsum
          mp = mpmax
        else:
          data = pm.find_MAP(**kwargs)
          mp = copy.deepcopy(data)
      elif method == 'none':
        data = None
        mp = self.hypers
      else:
        data = pm.sample(**kwargs)
        if method == 'mcmc_mean':
          mp = self.mean_extract(data)
        elif method == 'mcmc_map':
          mp = self.map_extract(data)
          try:
            mp = pm.find_MAP(start=mp)
          except:
            pass
        else:
          raise Exception('method must be one of map, mcmc_map, or mcmc_mean')

    # If method is none do not perform optimisation/sampling
    if method != 'none':
      # Input warping update
      if iwgp:
        self.iwgp_set(mp['iwgp'])

      # Output warping update
      if cwgp:
        rc = 0
        rcpos = 0
        params = []
        for i in range(self.yconrevs[0].np):
          if self.yconrevs[0].pos[i]:
            params.append(mp['cwgp_pos'][rcpos])
            rcpos += 1
          else:
            params.append(mp['cwgp'][rc])
            rc += 1
        self.cwgp_set(np.array(params))
        if iwgp:
          xin = np.zeros_like(x)
          for i in range(self.nx):
            xin[:,i] = self.xconrevs[i].con(x[:,i])
        yin = self.yconrevs[0].con(y[:,0])
        with m:
          gp = pm.gp.Marginal(cov_func=kern)
          y_ = gp.marginal_likelihood("y", X=xin, y=yin, sigma=pt.sqrt(gvar))
    else:
      if cwgp:
        if iwgp:
          xin = np.zeros_like(x)
          for i in range(self.nx):
            xin[:,i] = self.xconrevs[i].con(x[:,i])
        yin = self.yconrevs[0].con(y[:,0])
        with m:
          gp = pm.gp.Marginal(cov_func=kern)
          y_ = gp.marginal_likelihood("y", X=xin, y=yin, sigma=pt.sqrt(gvar))

    return m, gp, mp, data

  # Extract hyperparameter dictionary from means of mcmc data
  def mean_extract(self,data):
    mean = data.posterior.mean(dim=["chain", "draw"])
    mp = {}
    for key in mean:
      if len(mean[key].values.shape) > 1:
        mp[key] = mean[key].values
      else:
        mp[key] = np.array(mean[key].values)
    return mp

  # Extract hyperparameter dictionary from means of mcmc data
  def map_extract(self,data):
    pos = data.posterior.stack(draws=("chain", "draw"))
    ss = data.sample_stats.stack(draws=("chain", "draw"))
    lpmax = np.max(ss['lp'].values)
    if self.verbose:
      print(f'Max log posterior: {lpmax}')
    lpamax = np.argmax(ss['lp'].values)
    mp = {}
    for key in pos:
      if len(pos[key].values.shape) > 1:
        mp[key] = pos[key].values[:,lpamax]
      else:
        mp[key] = np.array(pos[key].values[lpamax])
    if self.verbose:
      print(f'Max log posterior sample: {mp}')
    return mp

  # Set output warping parameters
  def cwgp_set(self,params,mode='numpy',y=None):
    if y is None:
      y = self.y-self.ym
    warper = wgp(self.yconrevs[0].warping_names,params,y[:,0],mode=mode) 
    if mode == 'numpy':
      self.change_yconrevs([warper])
    else:
      return warper

  # Set input warping parameters
  def iwgp_set(self,params,mode='numpy',x=None):
    if x is None:
      x = self.x
    xconrevs = []
    rc = 0
    for i in range(self.nx):
      if isinstance(self.xconrevs[i],wgp):
        try:
          ran = len(self.xconrevs[i].params)
        except:
          ran = len(self.xconrevs[i].params.eval())
        xconrevs.append(wgp(self.xconrevs[i].warping_names,params[rc:rc+ran]\
            ,y=x[:,i],xdist=self.priors[i],mode=mode))
        rc += ran
      else:
        xconrevs.append(self.xconrevs[i])
    if mode == 'numpy':
      self.change_xconrevs(xconrevs=xconrevs)
    else:
      return xconrevs

  # Make train-test split and populate attributes
  def train_test(self,training_frac=0.9):
    self.nsamp = len(self.x)
    indexes = np.arange(self.nsamp)
    self.train,self.test = \
      train_test_split(indexes,train_size=training_frac)

  # Method to change noise/kernel attributes, scrubs any saved model
  def change_model(self,kernel=None,noise=None,mean=None):

    # Do nothing if argument not given
    if kernel is None:
      kernel = self.kernel
    if noise is None:
      noise = self.noise
    if mean is None:
      changed = False
    # Zero mean alias
    elif mean == 0:
      self.mean = self.zero_mean
      xm,ym = self._core__vector_solver(self.x,self.mean)
      if len(xm) != len(self.x):
        raise Exception("Mean function not valid at every x point in dataset")
      self.ym = ym
    # Assign and evaluate mean function
    else:
      self.mean = mean
      xm,ym = self._core__vector_solver(self.x,self.mean)
      if len(xm) != len(self.x):
        raise Exception("Mean function not valid at every x point in dataset")
      self.ym = ym

    # Split up kernel string for multi-kernel specs
    kerns = re.split(r'[+*]',kernel)
    ops = re.split(r'[ExponentialMatern32Matern52RBF]',kernel)
    ops[:] = [x for x in ops if x != '']

    # Check specified kernels are valid
    kernlist = ['RBF','Matern52','Matern32','Exponential','RatQuad']
    for i in kerns:
      if i not in kernlist:
        raise Exception(f"Error: kernel string must contain only {kernlist}")

    # Check validity of noise argument
    if not isinstance(noise,bool):
      raise Exception(f"Error: noise must be of type bool")

    # Assign attributes
    self.kernel = kernel
    self.kerns = kerns
    self.ops = ops
    self.nkern = len(kerns)
    self.noise = noise
    self.m = None
    self.gp = None
    self.hypers = None

  # Standard predict method which wraps the pymc predict
  def predict(self,x,return_var=False,convert=True,revert=True,normvar=False,jitter=1e-6,\
      EI=False,EIopt=None,deg=8):
    if convert:
      xarg = np.zeros_like(x)
      for i in range(self.nx):
        xarg[:,i] = self.xconrevs[i].con(x[:,i])
    else:
      xarg = copy.deepcopy(x)
      for i in range(self.nx):
        x[:,i] = self.xconrevs[i].rev(x[:,i])
    
    y, yv = self.__predict(self.m,self.gp,self.hypers,xarg,jitter)

    if revert:
      # Revert transforms
      y,yv = self.__gh_stats(x,y,yv,normvar,deg,EI=EI,EIopt=EIopt)

    if return_var:
      return y, yv
    else:
      return y

  # Get mean and variance of reverted variable by Gauss-Hermite quadrature
  def __gh_stats(self,x,y,yv,normvar=True,deg=8,EI=False,EIopt=None):

    # Reversion
    xi,wi = np.polynomial.hermite.hermgauss(deg)
    for i in range(len(y)):
      yi = np.sqrt(2*yv[i,0])*xi+y[i,0]
      yir = self.yconrevs[0].rev(yi)+self.mean(x[i,:])
      if EI:
        if EIopt == 'max':
          ydiff = yir-self.yopt
        else:
          ydiff = self.yopt-yir
        ydiff = np.where(ydiff > 0.0, ydiff, 0.0)
        y[i,0] = 1/np.sqrt(np.pi)*np.sum(wi*ydiff)
      else:
        y[i,0] = 1/np.sqrt(np.pi)*np.sum(wi*yir)
      yir2 = np.power(yir,2)
      ym2 = 1/np.sqrt(np.pi)*np.sum(wi*yir2)
      yv[i,0] = ym2-y[i,0]**2

    # Normalise variance output by mean squared
    if normvar:
      yv /= np.power(y,2)

    return y, yv

  # Get mean and variance of converted variable by Gauss-Hermite quadrature
  #TODO Make this and MCMC procedure compatible with non-zero mean function
  def __gh_stats_inv(self,y,yv,deg=8):

    # Reversion
    xi,wi = np.polynomial.hermite.hermgauss(deg)
    for i in range(len(y)):
      yi = np.sqrt(2*yv[i,0])*xi+y[i,0]
      yir = self.yconrevs[0].con(yi)
      ym = 1/np.sqrt(np.pi)*np.sum(wi*yir)
      yir2 = np.power(yir,2)
      ym2 = 1/np.sqrt(np.pi)*np.sum(wi*yir2)
      yvcon = ym2-ym**2

    return yvcon

  # Private predict method with more flexibility to act on any provided model
  def __predict(self,m,gp,hyps,x,jitter=1e-6):
    if self.verbose:
      print('Predicting...')
    t0 = stopwatch()
    with m:
      ypreds, yvarpreds = gp.predict(x, point=hyps,diag=True,\
          jitter=jitter,pred_noise=True)
    t1 = stopwatch()
    if self.verbose:
      print(f'Time taken: {t1-t0:0.2f} s')
    return ypreds.reshape((-1,1)),yvarpreds.reshape((-1,1))

  # Perform Bayesian optimisation
  def BO(self,opt_type='min',opt_method='predict',fit_method='map',max_iter=16,\
      method='EI',eps=0.1,iwgp=False,cwgp=False,jitter=1e-6,conv=0.01,\
      predict_samps=10000,normvar=True,refine=True,**kwargs):


    if self.ny > 1:
      raise Exception('Bayesian minimisation only implemented for single output')

    # Evaluate old optima
    if opt_type == 'max':
      xoptf = np.argmax
      yoptf = np.max
    elif opt_type == 'min':
      xoptf = np.argmin
      yoptf = np.min
    else:
      raise Exception('Error: opt_type argument must be one of max or min')
    self.xopt = self.x[xoptf(self.y[:,0]),:]
    self.yopt = yoptf(self.y)

    if self.verbose:
      print('Running Bayesian minimisation...')
      print(f'Current optima is {self.yopt} at x point {self.xopt}')

    if self.m is None:
      raise Exception('Model must be fitted before running Bayesian optimisation')

    if method == 'exploit':
      eps = 0.0

    # Get bounds from prior distributions
    lbs = np.zeros(self.nx)
    ubs = np.zeros(self.nx)
    for j in range(self.nx):
      lbs[j] = self.priors[j].ppf(1e-8)
      ubs[j] = self.priors[j].isf(1e-8)
    bnds = Bounds(lbs,ubs)

    # Iterate through optimisation algorithm
    xsampold = np.array([[1e300 for i in range(self.nx)]])
    for i in range(max_iter):

      if self.verbose:
        print(f'Iteration {i+1}')

      if opt_method == 'DE' or opt_method == 'predict':

        # Choose opt algorithm
        # Greedy eps random search or pure exploit (eps=0)
        if method == 'eps-RS' or method == 'exploit':
          def optf(x):
            if x.ndim == 1:
              x = np.array([x])
            ym = self.predict(x)
            if opt_type == 'min':
              return ym[:,0]
            else:
              return -ym[:,0]
        # Pure exploration
        elif method == 'explore':
          def optf(x):
            if x.ndim == 1:
              x = np.array([x])
            ym,yv = self.predict(x,return_var=True,normvar=normvar)
            return -yv[:,0]
        # Expected improvement
        elif method == 'EI':
          def optf(x):
            if x.ndim == 1:
              x = np.array([x])
            ym = self.predict(x,EI=True,EIopt = opt_type)
            return -ym[:,0]
        else:
          raise Exception('method must be one of eps-RS ,EI, exploit, or explore')

        # Opt or random if greedy search algorithm
        roll = np.random.rand()
        if method != 'eps-RS' or (method == 'eps-RS' and roll > eps):
          if opt_method == 'DE':
            # Differential evolution
            verb = self.verbose
            self.verbose = False
            res = differential_evolution(optf,bnds)
            self.verbose = verb
            xsamp = np.array([res.x])
            if self.verbose:
              print(f'Function opt is {res.fun:0.3f}')
          else:
            # LHC Sample and evaluate
            xsamps = self._LHC__latin_sample(predict_samps)
            ysamps = optf(xsamps)
            # Pick best
            xsamp = np.array([xsamps[np.argmin(ysamps),:]])
            if self.verbose:
              print(f'Function opt is {np.min(ysamps):0.3f}')
        else:
          xsamp = np.array([[j.rvs() for j in self.priors]])

      if not (opt_method == 'DE' or opt_method == 'predict') or \
          (opt_method == 'predict' and refine):
        # New mopt instance each iteration
        mopt = pm.Model()
        with mopt:

          # Convert distributions from scipy to pymc
          priors = []
          for k,j in enumerate(self.priors):
            if isinstance(j.dist,uniform_gen):
              intvl = j.support()
              prior = pm.Uniform(f'x{k}', lower=intvl[0], upper=intvl[1])
            elif isinstance(j.dist,norm_gen):
              prior = pm.Normal(f'x{k}', mu=j.mean(), sigma=j.std())
            elif isinstance(j.dist,truncnorm_gen):
              intvl = j.support()
              args = j.args
              kwds = j.kwds
              numkeys = len(kwds)
              numargs = len(args)
              if numargs == 4:
                mu = args[2]
                sigma = args[3]
              elif numargs == 3:
                mu = args[2]
                if 'scale' in kwds:
                  sigma = kwds['scale']
                else:
                  sigma = 1
              elif numkeys == 4:
                mu = kwds['loc']
                sigma = kwds['scale']
              elif numkeys+numargs == 2:
                mu = 0
                sigma = 1
              elif 'b' in kwds or numargs == 2:
                if 'loc' in kwds:
                  mu = kwds['loc']
                else:
                  mu = 0
                if 'scale' in kwds:
                  sigma = kwds['scale']
                else:
                  sigma = 1
              prior = pm.TruncatedNormal(f'x{k}',mu=mu,sigma=sigma, \
                  lower=intvl[0], upper=intvl[1])
            else:
              raise Exception('Prior distribution conversion from scipy to pymc not implemented')
            priors.append(prior)

          # Convert x inputs
          xin = pt.zeros((1,self.nx))
          for j in range(self.nx):
            xin = pt.set_subtensor(xin[0,j],\
                self.xconrevs[j].conmc(priors[j]))

          # Establish kernel
          for j in range(self.nkern):
            if self.kerns[j] == 'RBF':
              kerni = self.hypers['kv'][j]*pm.gp.cov.ExpQuad(self.nx,\
                  ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
            elif self.kerns[j] == 'RatQuad':
              # Only works if only one ratquad kernel specified 
              kerni = self.hypers['kv'][j]*pm.gp.cov.RatQuad(self.nx,alpha=self.hypers['alpha'],\
                  ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
            elif self.kerns[j] == 'Matern52':
              kerni = self.hypers['kv'][j]*pm.gp.cov.Matern52(self.nx,\
                  ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
            elif self.kerns[j] == 'Matern32':
              kerni = self.hypers['kv'][j]*pm.gp.cov.Matern32(self.nx,\
                  ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
            elif self.kerns[j] == 'Exponential':
              kerni = self.hypers['kv'][j]*pm.gp.cov.Exponential(self.nx,\
                  ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])

            # Apply kernel operations if more than one kern specified
            if j == 0:
              kern = kerni
            elif self.ops[j-1] == '+':
              kern += kerni
            elif self.ops[j-1] == '*':
              kern *= kerni

          # Build map to mean and variance predictions
          K = kern(self.xc)
          kstar = kern(self.xc,xin)
          if self.noise:
            K += pt.identity_like(K)*(jitter+self.hypers['gv'])
          else:
            K += pt.identity_like(K)*jitter
          L = pt.slinalg.cholesky(K)
          beta = pt.slinalg.solve_triangular(L,self.yc,lower=True)
          alpha = pt.slinalg.solve_triangular(L.T,beta)
          ycpmean = pt.sum(pt.dot(kstar.T,alpha))
          kstarstar = kern(xin)
          v = pt.slinalg.solve_triangular(L,kstar,lower=True)
          ycpvar = pt.sum(kstarstar-pt.dot(v.T,v))

          # Revert using Gauss quadrature
          xi,wi = np.polynomial.hermite.hermgauss(8)
          yi = pt.sqrt(2*ycpvar)*xi+ycpmean*pt.ones_like(xi)
          yir = self.yconrevs[0].revmc(yi)+self.mean(xin)
          ypmean = 1/np.sqrt(np.pi)*pt.dot(wi,yir)
          yir2 = pt.power(yir,2)
          ym2 = 1/np.sqrt(np.pi)*pt.dot(wi,yir2)
          ypvar = ym2-pt.power(ypmean,2)
          if normvar:
            ypvar /= ypmean**2

          # Expected improvement specific methods
          if method == 'EI':
            #ycoptm, ycoptv = \
            #    self.__predict(self.m,self.gp,self.hypers,np.array([self.xopt]))
            #yiopt = np.sqrt(2*ycoptv[:,0])*xi+ycoptm[:,0]
            #yioptr = self.yconrevs[0].rev(yiopt)+self.mean(self.xopt)
            if opt_type == 'max':
              ydiff = pt.maximum(pt.zeros_like(xi),yir-self.yopt*pt.ones_like(xi))
            else:
              ydiff = pt.maximum(pt.zeros_like(xi),self.yopt*pt.ones_like(xi)-yir)
            EI = 1/np.sqrt(np.pi)*pt.dot(wi,ydiff)

        # Choose opt algorithm
        # Greedy eps random search or pure exploit (eps=0)
        if method == 'eps-RS' or method == 'exploit':
          with mopt:
            # Maximise or minimise mean prediction
            if opt_type == 'max':
              y_ = pm.Potential('pot',ypmean)
            else:
              y_ = pm.Potential('pot',-ypmean)
        # Pure exploration
        elif method == 'explore':
          with mopt:
            # Maximise variance prediction
            y_ = pm.Potential('pot',ypvar)
        # Expected improvement
        elif method == 'EI':
          with mopt:
            # Maximise expected improvement
            y_ = pm.Potential('pot',EI)
        else:
          raise Exception('method must be one of eps-RS ,EI, exploit, or explore')

        roll = np.random.rand()
        if method != 'eps-RS' or (method == 'eps-RS' and roll > eps):
          with mopt:
            # Perform optimisation
            if opt_method == 'map' or (opt_method == 'predict' and refine):
              if opt_method == 'map': 
                start = {str(ky):np.random.normal() for ky in mopt.cont_vars}
                data = pm.find_MAP(start=start,**kwargs)
              else:
                for ix in range(len(priors)):
                  mopt.initial_values[mopt.free_RVs[ix]] = xsamp[0,ix]
                print(f'Refining {xsamp[0,:]}')
                data = pm.find_MAP(**kwargs)
              mp = copy.deepcopy(data)
              mpcheck = {str(ky):mp[str(ky)] for ky in mopt.cont_vars}
              print(mopt.point_logps(point=mpcheck))
            else:
              data = pm.sample(**kwargs)
              if opt_method == 'mcmc_mean':
                mp = self.mean_extract(data)
              elif opt_method == 'mcmc_map':
                mp = self.map_extract(data)
                try:
                  mp = pm.find_MAP(start=mp)
                except:
                  pass
              else:
                raise Exception(\
                    'opt_method must be one of map, mcmc_map, or mcmc_mean')

          # Extract sample from dictionary
          xsamp = np.zeros((1,self.nx))
          for j in range(self.nx):
            xsamp[0,j] = mp[f'x{j}']
        else:
          xsamp = np.array([[j.rvs() for j in self.priors]])

      # Evaluate convergence
      xdiff = np.sum(np.abs(xsamp-xsampold)/np.abs(xsampold))/self.nx
      if xdiff < conv:
        if self.verbose:
          print(\
              f'Convergence at relative tolerance {xdiff} achieved with point {xsamp}')
        break
      else:
        if self.verbose and i > 0:
          print(\
              f'Relative convergence in sample: {xdiff}')
        xsampold = xsamp

      # Check prediction at xsamp
      ypred = self.predict(xsamp)
      if self.verbose:
        print(f'Predicted {ypred} at x point {xsamp}')

      # Evaluate and update datasets
      xsamp,ysamp = self._core__vector_solver(xsamp)
      xm,ym = self._core__vector_solver(xsamp,self.mean)
      self.x = np.r_[self.x,xsamp]
      self.y = np.r_[self.y,ysamp]
      self.xc = np.r_[self.xc,self.__xconrev__(xsamp)]
      self.yc = np.r_[self.yc,self.__yconrev__(ysamp)]
      self.ym = np.r_[self.ym,ym]
      self.nsamp = len(self.x)
      
      if self.verbose:
        print(f'New sample is {ysamp+ym} at x point {xsamp}')

      # Evaluate optima
      self.xopt = self.x[xoptf(self.y[:,0]),:]
      self.yopt = yoptf(self.y)

      # Fit GP
      if fit_method == 'map':
        try:
          self.fit(method=fit_method,iwgp=iwgp,cwgp=cwgp,start=self.hypers)
        except:
          self.fit(method=fit_method,iwgp=iwgp,cwgp=cwgp)
      else:
        self.fit(method=fit_method,iwgp=iwgp,cwgp=cwgp)

    return self.xopt,self.yopt

  # y conversion shortcut
  def __yconrev__(self,yin,mode='con'):
    yout = np.zeros_like(yin)
    if mode == 'con':
      yout[:,0] = self.yconrevs[0].con(yin[:,0])
    elif mode == 'rev':
      yout[:,0] = self.yconrevs[0].rev(yin[:,0])
    else:
      raise Exception('Error: Mode must be one of con or rev')
    return yout

  # x conversion shortcut
  def __xconrev__(self,xin,mode='con'):
    xout = np.zeros_like(xin)
    for i in range(self.nx):
      if mode == 'con':
        xout[:,i] = self.xconrevs[i].con(xin[:,i])
      elif mode == 'rev':
        xout[:,i] = self.xconrevs[i].rev(xin[:,i])
      else:
        raise Exception('Error: Mode must be one of con or rev')
    return xout


  # Assess GP performance with several test plots and RMSE calcs
  def test_plots(self,revert=True,yplots=True,xplots=True,logscale=False\
      ,iwgp=False,cwgp=False,method='none',errorbars=True,saveyfig=None\
      ,xlab=None,ylab=None,returndat=False):
    # Creat train-test sets if none exist
    if self.train is None:
      self.train_test()
    xtrain = self.x[self.train,:]
    xtest = self.x[self.test,:]
    ytrain = self.y[self.train,:]
    ytest = self.y[self.test,:]
    ymtrain = self.ym[self.train,:]
    ymtest = self.ym[self.test,:]

    # Train model on training set and make predictions on xtest data
    m, gp, hypers, data = self.__fit(xtrain,ytrain-ymtrain,\
        method,iwgp,cwgp)
    #if self.verbose:
      #print(hypers)
    xctest = np.zeros_like(xtest)
    for i in range(self.nx):
      xctest[:,i] = self.xconrevs[i].con(xtest[:,i])
    ypred, yvars = self.__predict(m,gp,hypers,xctest)

    # Either revert data to original for comparison or leave as is
    if revert:
      ytest = ytest[:,0]
      ypred, yvars = self.__gh_stats(xtest,ypred,yvars,normvar=False)
      ypred = ypred[:,0]; yvars = yvars[:,0]
      meany = np.mean(self.y)
    else: 
      xtest = xctest
      ytest = self.yconrevs[0].con(ytest[:,0]-ymtest[:,0])
      meany = np.mean(self.yc)

    # RMSE for each y variable
    rmse = np.sqrt(np.mean(np.power(ypred-ytest,2)))
    mea = np.mean(np.abs(ypred-ytest))
    mpe = np.mean(np.abs(ypred-ytest)/np.abs(ytest))
    r2 = 1-np.sum(np.power(ypred-ytest,2))/np.sum(np.power(ytest-meany,2))
    if self.verbose:
      print(f'RMSE for y is: {rmse:0.5e}')
      print(f'Mean absoulte error for y is: {mea:0.5e}')
      print(f'Mean percentage error for y is: {mpe:0.5%}')
      print(f'R^2 for y is: {r2:0.5f}')
    # Compare ytest and predictions for each output variable
    if yplots:
      plt.plot(ytest,ytest,'-',label='True')
      if logscale:
        plt.plot(ytest,ypred,'o',label='GP')
        plt.xscale('log')
        plt.yscale('log')
      else:
        if errorbars:
          plt.errorbar(ytest,ypred,fmt='o',yerr=np.sqrt(yvars),\
              label='GP',capsize=3)
        else:
          plt.plot(ytest,ypred,'x',label='GP')
      if xlab is None:
        plt.xlabel(f'y')
      else:
        plt.xlabel(xlab)
      if ylab is None:
        plt.ylabel(f'y')
      else:
        plt.ylabel(ylab)
      plt.legend()
      if saveyfig is not None:
        plt.tight_layout()
        plt.savefig(saveyfig,bbox_inches='tight')
      else:
        plt.title(f'y')
      plt.show()
    # Compare ytest and predictions wrt each input variable
    if xplots:
      for j in range(self.nx):
        plt.title(f'y wrt x[{j}]')
        #xsort = np.sort(xtest[:,j])
        #asort = np.argsort(xtest[:,j])
        plt.plot(xtest[:,j],ytest,'.',label='Test')
        if logscale:
          plt.plot(xtest[:,j],ypred,'o',label='GP')
          plt.yscale('log')
        else:
          if errorbars:
            plt.errorbar(xtest[:,j],ypred,fmt='o',yerr=np.sqrt(yvars),\
                label='GP',capsize=3)
          else:
            plt.plot(xtest[:,j],ypred,'o',label='GP')
        plt.ylabel(f'y')
        plt.xlabel(f'x[{j}]')
        plt.legend()
        plt.show()

    if returndat:
      return xtest,ytest,ypred,yvars

  # Function which plots relative importances (inverse lengthscales)
  def relative_importances(self,logscale=False):

    if logscale:
      plt.bar([f'x[{i}]'for i in range(self.nx)],np.log(1/self.hypers['l']))
    else:
      plt.bar([f'x[{i}]'for i in range(self.nx)],1/self.hypers['l'])
    plt.ylabel('Relative importance')
    plt.show()

  # Fit GP with inverse optimisation
  def inverse_opt(self,yobs,yvarobs=None,\
      method='map',evaluate_opt=False,jitter=1e-6,**kwargs):

    if self.m is None:
      raise Exception('Model must be fitted before running Bayesian optimisation')

    if self.verbose:
      print('Running Bayesian inverse solver...')

    # Optimisation model
    mopt = pm.Model()
    with mopt:

      # Convert distributions from scipy to pymc
      priors = []
      for k,j in enumerate(self.priors):
        if isinstance(j.dist,uniform_gen):
          intvl = j.support()
          prior = pm.Uniform(f'x{k}', lower=intvl[0], upper=intvl[1])
        elif isinstance(j.dist,norm_gen):
          prior = pm.Normal(f'x{k}', mu=j.mean(), sigma=j.std())
        elif isinstance(j.dist,truncnorm_gen):
          intvl = j.support()
          args = j.args
          kwds = j.kwds
          numkeys = len(kwds)
          numargs = len(args)
          if numargs == 4:
            mu = args[2]
            sigma = args[3]
          elif numargs == 3:
            mu = args[2]
            if 'scale' in kwds:
              sigma = kwds['scale']
            else:
              sigma = 1
          elif numkeys == 4:
            mu = kwds['loc']
            sigma = kwds['scale']
          elif numkeys+numargs == 2:
            mu = 0
            sigma = 1
          elif 'b' in kwds or numargs == 2:
            if 'loc' in kwds:
              mu = kwds['loc']
            else:
              mu = 0
            if 'scale' in kwds:
              sigma = kwds['scale']
            else:
              sigma = 1
          prior = pm.TruncatedNormal(f'x{k}',mu=mu,sigma=sigma, \
              lower=intvl[0], upper=intvl[1])
        else:
          raise Exception('Prior distribution conversion from scipy to pymc not implemented')
        priors.append(prior)

      # Convert x inputs
      nobs = len(yobs)
      xin = pt.zeros((self.nsamp+nobs,self.nx))
      xin = pt.set_subtensor(xin[:-nobs,:],self.xc)
      for j in range(self.nx):
        xin = pt.set_subtensor(xin[-nobs:,j],\
            self.xconrevs[j].conmc(priors[j]))

      # Establish kernel
      for j in range(self.nkern):
        if self.kerns[j] == 'RBF':
          kerni = self.hypers['kv'][j]*pm.gp.cov.ExpQuad(self.nx,\
              ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
        elif self.kerns[j] == 'RatQuad':
          # Only works if only one ratquad kernel specified 
          kerni = self.hypers['kv'][j]*pm.gp.cov.RatQuad(self.nx,alpha=self.hypers['alpha'],\
              ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
        elif self.kerns[j] == 'Matern52':
          kerni = self.hypers['kv'][j]*pm.gp.cov.Matern52(self.nx,\
              ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
        elif self.kerns[j] == 'Matern32':
          kerni = self.hypers['kv'][j]*pm.gp.cov.Matern32(self.nx,\
              ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])
        elif self.kerns[j] == 'Exponential':
          kerni = self.hypers['kv'][j]*pm.gp.cov.Exponential(self.nx,\
              ls=self.hypers['l'][j*self.nx:(j+1)*self.nx])

        # Apply kernel operations if more than one kern specified
        if j == 0:
          kern = kerni
        elif self.ops[j-1] == '+':
          kern += kerni
        elif self.ops[j-1] == '*':
          kern *= kerni

      # Set y vector
      yin = np.zeros(self.nsamp+nobs)
      yin[:-nobs] = self.yc[:,0]
      yin[-nobs:] = self.yconrevs[0].con(yobs)

      # Set y noise vector
      ynoise = np.zeros(self.nsamp+nobs)
      if self.noise:
        ynoise[:-nobs] = np.sqrt(self.hypers['gv']+jitter)
      else:
        ynoise[:-nobs] = np.sqrt(jitter)
      if yvarobs is None:
        if self.noise:
          ynoise[:-nobs] = np.sqrt(self.hypers['gv']+jitter)
        else:
          ynoise[:-nobs] = np.sqrt(jitter)
      else:
        ynoise[-nobs:] = np.sqrt(self.__gh_stats_inv(yobs,yvarobs))

      # Get y derivative
      yfull = np.r_[self.y[:,0],yobs[:,0]]
      yder = self.yconrevs[0].der(yfull)

      # Evaluate likelihood
      nsamp = len(yin)
      K = kern(xin)
      K += pt.diag(ynoise)
      L = pt.slinalg.cholesky(K)
      beta = pt.slinalg.solve_triangular(L,yin,lower=True)
      alpha = pt.slinalg.solve_triangular(L.T,beta)
      y_ = pm.Potential('yopt',-0.5*pt.dot(yin.T,alpha)\
          -pt.sum(pt.log(pt.diag(L)))\
          -0.5*nsamp*pt.log(2*np.pi)\
          +pt.sum(pt.log(yder)))

      # Perform optimisation
      if method == 'map':
        start = {str(ky):np.random.normal() for ky in mopt.cont_vars}
        data = pm.find_MAP(start=start,**kwargs)
        mp = copy.deepcopy(data)
        mpcheck = {str(ky):mp[str(ky)] for ky in mopt.cont_vars}
        print(mopt.point_logps(point=mpcheck))
      else:
        data = pm.sample(**kwargs)
        if method == 'mcmc_mean':
          mp = self.mean_extract(data)
        elif method == 'mcmc_map':
          mp = self.map_extract(data)
          try:
            mp = pm.find_MAP(start=mp)
          except:
            pass
        else:
          raise Exception(\
              'method must be one of map, mcmc_map, or mcmc_mean')

    # Extract sample from dictionary
    xopt = np.zeros((1,self.nx))
    for j in range(self.nx):
      xopt[0,j] = mp[f'x{j}']

    # Check prediction at xsamp
    ypred = self.predict(xopt)
    if self.verbose:
      print(f'Predicted {ypred} at x point {xopt}')

    if evaluate_opt:
      # Evaluate and update datasets
      xsamp,ysamp = self._core__vector_solver(xopt)
      xm,ym = self._core__vector_solver(xopt,self.mean)
      self.x = np.r_[self.x,xsamp]
      self.y = np.r_[self.y,ysamp]
      self.xc = np.r_[self.xc,self.__xconrev__(xsamp)]
      self.yc = np.r_[self.yc,self.__yconrev__(ysamp)]
      self.ym = np.r_[self.ym,ym]
      self.nsamp = len(self.x)
      
      if self.verbose:
        print(f'Actual evaluation is {ysamp+ym} at x point {xsamp}')

      xopt = xopt[0,:]
      ysamp = ysamp[0]
      return data, xopt, ysamp
    else:
      xopt = xopt[0,:]
      return data, xopt
