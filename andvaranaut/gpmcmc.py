#!/bin/python3

import numpy as np
from design import latin_random,ihs
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
import ray
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy.optimize import Bounds
from scipy.stats._continuous_distns import uniform_gen, norm_gen
import re

# Zero mean function
def zero_mean(x):
  ret = np.array([0.0])
  return ret

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
    self.ym = np.empty((0,1))

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
  def y_dist(self,mode='hist_kde',nsamps=None,return_data=False,surrogate=True):
    # Allow for use of surrogate evaluations or underlying datasets
    if surrogate:
      xsamps = super()._LHC__latin_sample(nsamps)
      xcons = np.zeros((nsamps,self.nx))
      for i in range(self.nx):
        xcons[:,i] = self.xconrevs[i].con(xsamps[:,i])
      ypreds = self.predict(xcons)
      yrevs = np.zeros((nsamps,self.ny))
      for i in range(self.ny):
        yrevs[:,i] = self.yconrevs[i].rev(ypreds[:,i])
      amax = np.argmax(ypreds)
      idx = (amax//self.ny,amax%self.ny)
      super()._LHC__y_dist(yrevs,mode)
      if return_data:
        return xsamps,yrevs
    elif not surrogate:
      super().y_dist(mode)
    else:
      raise Exception("Error: surrogate argument must be of type bool")

  def __scrub_train_test(self):
    self.train = None
    self.test = None

  # Sample method inherits lhc sampling and adds adaptive sampling option
  def sample(self,nsamps,seed=None):
    # Evaluate samples
    super().sample(nsamps,seed)

    # Evaluate mean function
    xm,ym = self._core__vector_solver(self.x,self.mean)
    if len(xm) != len(self.x):
      raise Exception("Mean function not valid at every x point in dataset")
    self.ym = np.r_[self.ym,ym]

    # Get converted data
    self.__con(len(self.x))

  # Fit GP standard method
  def fit(self,method='map',return_data=False,\
      iwgp=False,cwgp=False,jitter=1e-6,truncate=False,**kwargs):
    self.m, self.gp, self.hypers, data = self.__fit(self.x,self.y-self.ym,\
        method,iwgp,cwgp,jitter,truncate,**kwargs)

    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,method,iwgp,cwgp,jitter=1e-6,truncate=False,**kwargs):
    
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
        y_ = pm.Potential('ypot',-0.5*pt.dot(yin.T,alpha)\
            -pt.sum(pt.log(pt.diag(L)))\
            -0.5*nsamp*pt.log(2*np.pi)\
            +pt.sum(pt.log(yder)))
      else:
        gp = pm.gp.Marginal(cov_func=kern)
        y_ = gp.marginal_likelihood("y", X=xin, y=yin, sigma=pt.sqrt(gvar), \
            jitter=jitter)

      # Fit and process results depending on method
      if method == 'map':
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
      self.mean = zero_mean
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

  # Standard predict method which wraps the GPy predict and allows for parallelism
  def predict(self,x,return_var=False,convert=True,revert=True,normvar=True,jitter=1e-6):
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
      y,yv = self.__gh_stats(x,y,yv,normvar)

    if return_var:
      return y, yv
    else:
      return y

  # Get mean and variance of reverted variable by Gauss-Hermite quadrature
  def __gh_stats(self,x,y,yv,normvar=True,deg=8):

    # Reversion
    xi,wi = np.polynomial.hermite.hermgauss(deg)
    for i in range(len(y)):
      yi = np.sqrt(2*yv[i,0])*xi+y[i,0]
      yir = self.yconrevs[0].rev(yi)+self.mean(x[i,:])
      yir2 = np.power(yir,2)
      y[i,0] = 1/np.sqrt(np.pi)*np.sum(wi*yir)
      ym2 = 1/np.sqrt(np.pi)*np.sum(wi*yir2)
      yv[i,0] = ym2-y[i,0]**2

    # Normalise variance output by mean squared
    if normvar:
      yv /= np.power(y,2)

    return y, yv

  # Private predict method with more flexibility to act on any provided GPy model
  def __predict(self,m,gp,hyps,x,jitter=1e-6):
    if self.verbose:
      print('Predicting...')
    t0 = stopwatch()
    if self.parallel:
      # Break x values into maximal sized chunks and use ray parallelism
      if not ray.is_initialized():
        ray.init(num_cpus=self.nproc,log_to_driver=False)
      chunks = np.minimum(len(x),self.nproc)
      crem = len(x) % chunks
      csize = round((len(x)-crem)/chunks)
      csizes = np.ones(chunks,dtype=np.intc)*csize
      csizes[:crem] += 1
      cinds = np.array([np.sum(csizes[:i+1]) for i in range(chunks)])
      cinds = np.r_[0,cinds]
      gpred = lambda x: gp.predict(x, point=hyps,  diag=True)
      outs = ray.get([_parallel_predict.remote(\
          x[cinds[i]:cinds[i+1]].reshape((csizes[i],self.nx)),\
          gpred) for i in range(chunks)]) # Chunks
      ypreds = np.empty((0,self.ny)); yvarpreds = np.empty((0,outs[0][1].shape[1]))
      for i in range(chunks):
        ypreds = np.r_[ypreds,outs[i][0]]
        yvarpreds = np.r_[yvarpreds,outs[i][1]]
      #ray.shutdown()
    else:
      with m:
        ypreds, yvarpreds = gp.predict(x, point=hyps,  diag=True,jitter=jitter)
    t1 = stopwatch()
    if self.verbose:
      print(f'Time taken: {t1-t0:0.2f} s')
    return ypreds.reshape((-1,1)),yvarpreds.reshape((-1,1))

  # Minimise output variable using GPyOpt 
  def bayesian_minimisation(self,max_iter=15,minmax=False,restarts=5,revert=True):

    if self.ny > 1:
      raise Exception('Bayesian minimisation only implemented for single output')

    if self.verbose:
      print('Running Bayesian minimisation...')
    
    # Setup domain as list of dictionaries for each variable
    lbs = np.zeros(self.nx)
    ubs = np.zeros(self.nx)
    tiny = np.finfo(np.float64).tiny
    for i in range(self.nx):
      if minmax:
        lbs[i] = np.min(self.xc[:,i])
        ubs[i] = np.max(self.xc[:,i])
      else:
        lbs[i] = self.xconrevs[i].con(self.priors[i].ppf(tiny))
        ubs[i] = self.xconrevs[i].con(self.priors[i].isf(tiny))
    bnds = Bounds(lb=lbs,ub=ubs)

    ## TODO: Constraint setup
    
    # Setup objective function which is target plus conversion/reversion
    def target_transform(xc):
      xr = np.zeros_like(xc)
      for i in range(self.nx):
        xr[i] = self.xconrevs[i].rev(xc[i])
      xr,yr = self._core__vector_solver(np.array([xr]))
      xm,ym = self._core__vector_solver(np.array([xr]),self.mean)
      yc = self.yconrevs[0].con(yr[:,0]-ym[:,0])
      # Update database with new BO samples
      self.x = np.r_[self.x,xr]
      self.y = np.r_[self.y,yr]
      self.xc = np.r_[self.xc,np.array([xc])]
      self.yc = np.r_[self.yc,np.array([yc])]
      self.ym = np.r_[self.ym,ym]
      self.nsamp = len(self.x)
      return yc

    # Run optimisation
    try:
      res = self._core__opt(target_transform,'BO',self.nx,restarts,\
          bounds=bnds,max_iter=max_iter)
      xopt = res.x
      yopt = res.fun

      # Revert to original data
      if revert:
        yopt = self.yconrevs[0].rev(yopt)+self.mean(xopt)
        for i in range(self.nx):
          xopt[i] = self.xconrevs[i].rev(xopt[i])

    except:
      print('Warning: Bayesian minimisation failed, choosing best sample in dataset')
      yopt = np.min(self.y)
      xopt = self.x[np.argmin(self.y[:,0]),:]


    self.xopt = xopt
    self.yopt = yopt

    return xopt,yopt

  # Perform Bayesian optimisation
  def BO(self,opt_type='min',opt_method='map',fit_method='map',max_iter=15,\
      method='eps-RS',eps=0.1,iwgp=False,cwgp=False,jitter=1e-6,**kwargs):

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
    xopt = self.x[xoptf(self.y[:,0]),:]
    yopt = yoptf(self.y)

    if self.verbose:
      print('Running Bayesian minimisation...')
      print(f'Current optima is {yopt} at x point {xopt}')

    if self.m is None:
      raise Exception('Model must be fitted before running Bayesian optimisation')

    mopt = pm.Model()
    
    with mopt:

      # Convert distributions from scipy to pymc
      priors = []
      for i,j in enumerate(self.priors):
        if isinstance(j.dist,uniform_gen):
          if len(j.args) >= 2:
            prior = pm.Uniform(f'x{i}',lower=j.args[0],\
                upper=j.args[0]+j.args[1])
          elif len(j.args) == 1:
            prior = pm.Uniform(f'x{i}',lower=j.args[0],\
                upper=j.args[0]+j.kwds['scale'])
          else:
            prior = pm.Uniform(f'x{i}',lower=j.kwds['loc'],\
                upper=j.kwds['loc']+j.kwds['scale'])
        elif isinstance(j.dist,norm_gen):
          if len(j.args) >= 2:
            prior = pm.Normal(f'x{i}',mu=j.args[0],\
                sigma=j.args[1])
          elif len(j.args) == 1:
            prior = pm.Normal(f'x{i}',mu=j.args[0],\
                sigma=j.kwds['scale'])
          else:
            prior = pm.Normal(f'x{i}',mu=j.kwds['loc'],\
                sigma=j.kwds['scale'])
        else:
          raise Exception('Prior distribution conversion from scipy to pymc not implemented')
        priors.append(prior)

      # Convert x inputs
      xin = pt.zeros((1,self.nx))
      for j in range(self.nx):
        xin = pt.set_subtensor(xin[0,j],\
            self.xconrevs[j].conmc(priors[j]))

      # Establish kernel
      for i in range(self.nkern):
        if self.kerns[i] == 'RBF':
          kerni = self.hypers['kv'][i]*pm.gp.cov.ExpQuad(self.nx,\
              ls=self.hypers['l'][i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'RatQuad':
          # Only works if only one ratquad kernel specified 
          kerni = self.hypers['kv'][i]*pm.gp.cov.RatQuad(self.nx,alpha=self.hypers['alpha'],\
              ls=self.hypers['l'][i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'Matern52':
          kerni = self.hypers['kv'][i]*pm.gp.cov.Matern52(self.nx,\
              ls=self.hypers['l'][i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'Matern32':
          kerni = self.hypers['kv'][i]*pm.gp.cov.Matern32(self.nx,\
              ls=self.hypers['l'][i*self.nx:(i+1)*self.nx])
        elif self.kerns[i] == 'Exponential':
          kerni = self.hypers['kv'][i]*pm.gp.cov.Exponential(self.nx,\
              ls=self.hypers['l'][i*self.nx:(i+1)*self.nx])

        # Apply kernel operations if more than one kern specified
        if i == 0:
          kern = kerni
        elif self.ops[i-1] == '+':
          kern += kerni
        elif self.ops[i-1] == '*':
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
      yir2 = pt.power(yir,2)
      ypmean = 1/np.sqrt(np.pi)*pt.dot(wi,yir)
      ym2 = 1/np.sqrt(np.pi)*pt.dot(wi,yir2)
      ypvar = ym2-pt.power(ypmean,2)

    # Iterate through optimisation algorithm
    for i in range(max_iter):

      # Choose opt algorithm
      # Greedy eps random search
      if method == 'eps-RS':

        # Roll for opt mean or random search
        roll = np.random.rand()
        if roll > eps:
          with mopt:
            # Maximise or minimise mean prediction
            if i == 0:
              if opt_type == 'max':
                y_ = pm.Potential('pot',ypmean)
              else:
                y_ = pm.Potential('pot',-ypmean)

            # Perform optimisation
            if opt_method == 'map':
              data = pm.find_MAP(**kwargs)
              mp = copy.deepcopy(data)
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
        self.fit(method=fit_method,iwgp=iwgp,cwgp=cwgp,start=self.hypers)
      else:
        self.fit(method=fit_method,iwgp=iwgp,cwgp=cwgp)

    return xopt,yopt

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
    if self.verbose:
      print(hypers)
    xctest = np.zeros_like(xtest)
    for i in range(self.nx):
      xctest[:,i] = self.xconrevs[i].con(xtest[:,i])
    ypred, yvars = self.__predict(m,gp,hypers,xctest)

    # Either revert data to original for comparison or leave as is
    if revert:
      ytest = ytest[:,0]
      ypred, yvars = self.__gh_stats(xtest,ypred,yvars,normvar=False)
      ypred = ypred[:,0]; yvars = yvars[:,0]
    else: 
      xtest = xctest
      ytest = self.yconrevs[0].con(ytest[:,0]-ymtest[:,0])

    # RMSE for each y variable
    rmse = np.sqrt(np.mean(np.power(ypred-ytest,2)))
    mea = np.mean(np.abs(ypred-ytest))
    mpe = np.mean(np.abs(ypred-ytest)/np.abs(ytest))
    if self.verbose:
      print(f'RMSE for y is: {rmse:0.5e}')
      print(f'Mean absoulte error for y is: {mea:0.5e}')
      print(f'Mean percentage error for y is: {mpe:0.5%}')
    # Compare ytest and predictions for each output variable
    if yplots:
      plt.plot(ytest,ytest,'-',label='True')
      if logscale:
        plt.plot(ytest,ypred,'x',label='GP')
        plt.xscale('log')
        plt.yscale('log')
      else:
        if errorbars:
          plt.errorbar(ytest,ypred,fmt='x',yerr=np.sqrt(yvars),\
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
          plt.plot(xtest[:,j],ypred,'x',label='GP')
          plt.yscale('log')
        else:
          if errorbars:
            plt.errorbar(xtest[:,j],ypred,fmt='x',yerr=np.sqrt(yvars),\
                label='GP',capsize=3)
          else:
            plt.plot(xtest[:,j],ypred,'x',label='GP')
        plt.ylabel(f'y')
        plt.xlabel(f'x[{j}]')
        plt.legend()
        plt.show()

    if returndat:
      return xtest,ytest,ypred,yvars

  """
  # Function which plots relative importances (inverse lengthscales)
  def relative_importances(self,original_data=False,restarts=10,scale='std_dev'):
    # Check scale arg
    scales = ['mean','std_dev']
    if scale not in scales:
        raise Exception(f"Error: scale must be one of {scales}")
    # Get means and std deviations for scaling
    means = np.array([self.priors[i].mean() for i in range(self.nx)])
    stds = np.array([self.priors[i].std() for i in range(self.nx)])
    # Choose whether to train GP on original or converted data
    if original_data:
      m = self.__fit(self.x,self.y,restarts=restarts)
    else:
      m = self.m
      means = np.array([self.xconrevs[i].con(means[i]) for i in range(self.nx)])
      stds = np.array([np.std(self.xc[:,i]) for i in range(self.nx)])
    # Select scale
    if scale == 'mean':
        facs = means
    else:
        facs = stds
    # Evaluate importances
    sens_gp = np.zeros(self.nx)
    for i in range(self.nx):
      leng = m.kern.lengthscale[i]
      if facs[i] == 0:
          facs[i] = 1
      sens_gp[i] = facs[i]/leng
    plt.bar([f'x[{i}]'for i in range(self.nx)],np.log(sens_gp))
    plt.ylabel('Relative log importance')
    plt.show()

  def portable_save(self,fname):
    pg = copy.deepcopy(self)
    pg.target = None
    pg.constraints = None
    save_object(pg,fname)

  # Fit GP with inverse optimisation
  def inverse_opt(self,yobs,\
      method='mcmc_mean',return_data=False,iwgp=False,**kwargs):
    self.m, self.gp, self.hypers, data = self.__inverse(yobs,method,\
        iwgp,**kwargs)
    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __inverse(self,yobs,method,iwgp,**kwargs):
    
    # PyMC context manager
    m = pm.Model()
    with m:
      # Priors on GP hyperparameters
      if self.noise:
        gvar = pm.HalfNormal('gv',sigma=0.4)
      else:
        gvar = 1e-8
      kls = pm.Gamma('l',alpha=2.15,beta=6.91,shape=self.nx)
      kvar = pm.Gamma('kv',alpha=4.3,beta=5.3)

      # Add y observations to existing dataset
      obs = len(yobs)
      yobc = self.yconrevs[0].con(yobs)
      yin = np.vstack([self.yc,yobc]).flatten()

      # Convert distributions from scipy to pymc
      priors = []
      for i,j in enumerate(self.priors):
        if isinstance(j.dist,uniform_gen):
          if len(j.args) >= 2:
            prior = pm.Uniform(f'x{i}',lower=j.args[0],\
                upper=j.args[0]+j.args[1])
          elif len(j.args) == 1:
            prior = pm.Uniform(f'x{i}',lower=j.args[0],\
                upper=j.args[0]+j.kwds['scale'])
          else:
            prior = pm.Uniform(f'x{i}',lower=j.kwds['loc'],\
                upper=j.kwds['loc']+j.kwds['scale'])
        elif isinstance(j.dist,norm_gen):
          if len(j.args) >= 2:
            prior = pm.Normal(f'x{i}',mu=j.args[0],\
                sigma=j.args[1])
          elif len(j.args) == 1:
            prior = pm.Normal(f'x{i}',mu=j.args[0],\
                sigma=j.kwds['scale'])
          else:
            prior = pm.Normal(f'x{i}',mu=j.kwds['loc'],\
                sigma=j.kwds['scale'])
        else:
          raise Exception('Prior distribution conversion from scipy to pymc not implemented')
        priors.append(prior)

      # Input warping
      xin = pt.zeros((self.nsamp+obs,self.nx))
      if iwgp:
        rc = 0
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            rc += self.xconrevs[i].np
        iwgp = pm.Gamma('iwgp',alpha=4.3,beta=5.3,shape=rc)
        rc = 0
        check = True
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            check = False
            rvs = [iwgp[k] for k in range(rc,rc+self.xconrevs[i].np)]
            xin = pt.set_subtensor(xin[:self.nsamp,i],\
                self.xconrevs[i].conmc(self.x[:,i],rvs))
            xin = pt.set_subtensor(xin[self.nsamp:,i],\
                self.xconrevs[i].conmc(priors[i],rvs))
            rc += self.xconrevs[i].np
          else:
            xin = pt.set_subtensor(xin[:self.nsamp,i],\
                self.xconrevs[i].conmc(self.x[:,i],np.empty(0)))
            xin = pt.set_subtensor(xin[self.nsamp:,i],\
                self.xconrevs[i].conmc(priors[i],np.empty(0)))
        if check:
          raise Exception('Error: iwgp set to true but none of xconrevs are wgp classes')
      else:
        xin = pt.set_subtensor(xin[:self.nsamp],self.xc)
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            xin = pt.set_subtensor(xin[self.nsamp:,i],\
                self.xconrevs[i].conmc(priors[i],\
                pt.as_tensor_variable(self.xconrevs[i].params)))
          else:
            xin = pt.set_subtensor(xin[self.nsamp:,i],\
                self.xconrevs[i].conmc(priors[i],np.empty(0)))

      # Setup kernel
      if self.kernel == 'RBF':
        kern = kvar*pm.gp.cov.ExpQuad(self.nx,ls=kls)
      elif self.kernel == 'Matern52':
        kern = kvar*pm.gp.cov.Matern52(self.nx,ls=kls)
      elif self.kernel == 'Matern32':
        kern = kvar*pm.gp.cov.Matern32(self.nx,ls=kls)
      elif self.kernel == 'Exponential':
        kern = kvar*pm.gp.cov.Exponential(self.nx,ls=kls)

      # GP and likelihood
      gp = pm.gp.Marginal(cov_func=kern)
      y_ = gp.marginal_likelihood("y", X=xin, y=yin, noise=gvar)

      # Fit and process results depending on method
      if method == 'map':
        data = pm.find_MAP(**kwargs)
        mp = copy.deepcopy(data)
      else:
        data = pm.sample(**kwargs)
        if method == 'mcmc_mean':
          mp = self.mean_extract(data)
        elif method == 'mcmc_map':
          mp = self.map_extract(data)
        else:
          raise Exception('method must be one of map, mcmc_map, or mcmc_mean')

      mp['x'] = np.array([mp['x0'],mp['x1'],mp['x2']])
      self.xopt = mp['x']

      # Input warping update
      if iwgp:
        self.iwgp_set(mp['iwgp'])

    return m, gp, mp, data
  """
