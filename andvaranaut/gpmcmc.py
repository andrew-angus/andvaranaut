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
from andvaranaut.lhc import _surrogate
from andvaranaut.transform import wgp
import ray
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# Inherit from surrogate class and add GP specific methods
class GPMCMC(_surrogate):
  def __init__(self,kernel='RBF',noise=True,prior_dict=None,**kwargs):
    super().__init__(**kwargs)
    self.change_model(kernel,noise)
    self.set_prior_dict(prior_dict)
    self.__scrub_train_test()

  # Set prior dictionary, defaulting to lognormals on hypers
  def set_prior_dict(self,prior_dict):
    if prior_dict is None:
      prior_dict = {}
      if self.noise:
        prior_dict['hypers'] = [st.norm() for i in range(5)]
      else:
        prior_dict['hypers'] = [st.norm() for i in range(4)]
    self.prior_dict = prior_dict

  def __scrub_train_test(self):
    self.xtrain = None
    self.xtest = None
    self.ytrain = None
    self.ytest = None
    self.train = None
    self.test = None

  # Sample method inherits lhc sampling and adds adaptive sampling option
  def sample(self,nsamps,method='lhc',batchsize=1,restarts=10,\
      seed=None,opt_method='DE',opt_restarts=10):
    methods = ['lhc','adaptive']
    if method not in methods:
      raise Exception(f'Error: method must be one of {methods}')
    if method == 'lhc':
      super().sample(nsamps,seed)
    else:
      raise Exception('No other method implemented, choose LHC')

  # Inherit del_samples and extend to remove test-train datasets
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    super().del_samples(ndels,method,idx)
    self.__scrub_train_test()

  # Fit GP standard method
  def fit(self,method='mcmc_mean',return_data=False,iwgp=False,**kwargs):
    self.m, self.gp, self.hypers, data = self.__fit(self.xc,self.yc,method,\
        iwgp,**kwargs)
    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,method,iwgp,**kwargs):
    
    # PyMC context manager
    m = pm.Model()
    with m:
      # Priors on GP hyperparameters
      if self.noise:
        gvar = pm.HalfNormal('gv',sigma=0.4)
      else:
        gvar = 1e-8
      kls = pm.Gamma('l',alpha=2.15,beta=6.91,shape=3)
      kvar = pm.Gamma('kv',alpha=4.3,beta=5.3)

      # Input warping
      if iwgp:
        rc = 0
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            rc += self.xconrevs[i].np
        iwgp = pm.Gamma('iwgp',alpha=4.3,beta=5.3,shape=rc)
        rc = 0
        x1 = []
        check = True
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            check = False
            rvs = [iwgp[k] for k in range(rc,rc+self.xconrevs[i].np)]
            x1.append(self.xconrevs[i].conmc(self.x[:,i],rvs))
            rc += self.xconrevs[i].np
          else:
            x1.append(self.xc[:,i])
          xin = pt.stack(x1,axis=1)
        if check:
          raise Exception('Error: iwgp set to true but none of xconrevs are wgp classes')
      else:
        xin = self.xc

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
      y_ = gp.marginal_likelihood("y", X=xin, y=self.yc[:,0], noise=gvar)

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

      # Input warping update
      if iwgp:
        self.iwgp_set(mp['iwgp'])

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
    return mp

  # Use modified Gaussian likelihood to fit cwgp parameters
  def y_warp(self,method='mcmc_map',return_data=True,**kwargs):

    # PyMC context manager
    m = pm.Model()
    with m:

      # Output warping
      yc = self.yconrevs[0]
      if not isinstance(yc,wgp):
        raise Exception('Error: cwgp set to true but yconrevs class is not wgp')
      npar = yc.np
      if npar == 0:
        raise Exception('Error: cwgp set to true but wgp class has no tuneable parameters')
      rc = 0
      rcpos = 0
      for i in range(npar):
        if yc.pos[i]:
          rcpos += 1
        else:
          rc += 1
      cwgpp = pm.Gamma('cwgp_pos',alpha=4.3,beta=5.3,shape=rcpos)
      cwgp = pm.Normal('cwgp',mu=0.0,sigma=1.0,shape=rc)
      rc = 0
      rcpos = 0
      rvs = []
      for i in range(npar):
        if yc.pos[i]:
          rvs.append(cwgpp[rcpos])
          rcpos += 1
        else:
          rvs.append(cwgp[rc])
          rc += 1
      yin = yc.conmc(self.y[:,0],rvs)

      # GP and likelihood
      y_ = pm.Normal('yobs',mu=yin,sigma=1.0,observed=np.zeros(self.nsamp)) 

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

      # Output warping update
      rc = 0
      rcpos = 0
      params = []
      for i in range(npar):
        if yc.pos[i]:
          params.append(mp['cwgp_pos'][rcpos])
          rcpos += 1
        else:
          params.append(mp['cwgp'][rc])
          rc += 1
      self.cwgp_set(np.array(params))

    return mp, data


  # Set output warping parameters
  def cwgp_set(self,x):
    self.change_yconrevs([wgp(self.yconrevs[0].warping_names,x,self.y)])

  # Set input warping parameters
  def iwgp_set(self,x):
    xconrevs = []
    rc = 0
    for i in range(self.nx):
      if isinstance(self.xconrevs[i],wgp):
        ran = len(self.xconrevs[i].params)
        xconrevs.append(wgp(self.xconrevs[i].warping_names,x[rc:rc+ran]\
            ,y=self.x[:,i],xdist=self.priors[i]))
        rc += ran
      else:
        xconrevs.append(self.xconrevs[i])
    self.change_xconrevs(xconrevs=xconrevs)

  # Make train-test split and populate attributes
  def train_test(self,training_frac=0.9):
    indexes = np.arange(self.nsamp)
    self.train,self.test = \
      train_test_split(indexes,train_size=training_frac)

  # Method to change noise/kernel attributes, scrubs any saved model
  def change_model(self,kernel,noise):
    kerns = ['RBF','Matern52','Matern32','Exponential']
    if kernel not in kerns:
      raise Exception(f"Error: kernel must be one of {kerns}")
    if not isinstance(noise,bool):
      raise Exception(f"Error: noise must be of type bool")
    self.kernel = kernel
    self.noise = noise
    self.m = None
    self.gp = None
    self.hypers = None

  # Inherit set_data method and scrub train-test sets
  def set_data(self,x,y):
    super().set_data(x,y)
    self.__scrub_train_test()

  # Inherit and extend y_dist to have dist by surrogate predictions
  def y_dist(self,mode='hist_kde',nsamps=None,return_data=False,surrogate=True):
    return super().y_dist(mode,nsamps,return_data,surrogate,self.predict)

  # Standard predict method which wraps the GPy predict and allows for parallelism
  def predict(self,x,return_var=False,convert=True,revert=True):
    if convert:
      xarg = np.zeros_like(x)
      for i in range(self.nx):
        xarg[:,i] = self.xconrevs[i].con(x[:,i])
    else:
      xarg = copy.deepcopy(x)
    
    y = self.__predict(self.gp,self.hypers,xarg,return_var)

    if revert and not return_var:
      for i in range(self.ny):
        y[:,i] = self.yconrevs[i].rev(y[:,i])
    elif revert:
      raise Exception("Reversion of variance not implemented, set return_var = False")

    return y

  # Private predict method with more flexibility to act on any provided GPy model
  def __predict(self,gp,hyps,x,return_var=False):
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
      with self.m:
        ypreds, yvarpreds = gp.predict(x, point=hyps,  diag=True)
    t1 = stopwatch()
    if self.verbose:
      print(f'Time taken: {t1-t0:0.2f} s')
    if return_var:
      return ypreds.reshape((-1,1)),yvarpreds.reshape((-1,1))
    else:
      return ypreds.reshape((-1,1))

  """
  # Assess GP performance with several test plots and RMSE calcs
  def test_plots(self,restarts=10,revert=True,yplots=True,xplots=True,opt=True):
    # Creat train-test sets if none exist
    #if self.xtrain is None:
    if self.train is None:
      self.train_test()
    self.xtrain = self.xc[self.train,:]
    self.xtest = self.xc[self.test,:]
    self.ytrain = self.yc[self.train,:]
    self.ytest = self.yc[self.test,:]
    # Train model on training set and make predictions on xtest data
    if self.m is None:
      mtrain = self.__fit(self.xtrain,self.ytrain,restarts=restarts,opt=opt)
    else:
      mtrain = copy.deepcopy(self.m)
      mtrain.set_XY(self.xtrain,self.ytrain)
      if opt:
        mtrain.optimize_restarts(restarts,robust=True,verbose=self.verbose)
        #mtrain.kern.lengthscale = self.m.kern.lengthscale
        #mtrain.kern.variance = self.m.kern.variance
        #mtrain.Gaussian_noise.variance = self.m.Gaussian_noise.variance
    #ypred,ypred_var = mtrain.predict(self.xtest)
    ypred = self.__predict(mtrain,self.xtest,return_var=False)
    # Either revert data to original for comparison or leave as is
    if revert:
      xtest = np.zeros_like(self.xtest)
      ytest = np.zeros_like(self.ytest)
      for i in range(self.nx):
        xtest[:,i] = self.xconrevs[i].rev(self.xtest[:,i])
      for i in range(self.ny):
        ypred[:,i] = self.yconrevs[i].rev(ypred[:,i])
        ytest[:,i] = self.yconrevs[i].rev(self.ytest[:,i])
    else: 
      xtest = self.xtest
      ytest = self.ytest
    # RMSE for each y variable
    #print('')
    for i in range(self.ny):
      rmse = np.sqrt(np.sum((ypred[:,i]-ytest[:,i])**2)/len(ytest[:,i]))
      if self.verbose:
        print(f'RMSE for y[{i}] is: {rmse}')
    # Compare ytest and predictions for each output variable
    if yplots:
      for i in range(self.ny):
        plt.title(f'y[{i}]')
        plt.plot(ytest[:,i],ytest[:,i],'-',label='True')
        plt.plot(ytest[:,i],ypred[:,i],'X',label='GP')
        plt.ylabel(f'y[{i}]')
        plt.xlabel(f'y[{i}]')
        plt.legend()
        plt.show()
    # Compare ytest and predictions wrt each input variable
    if xplots:
      for i in range(self.ny):
        for j in range(self.nx):
          plt.title(f'y[{i}] wrt x[{j}]')
          xsort = np.sort(xtest[:,j])
          asort = np.argsort(xtest[:,j])
          plt.plot(xsort,ypred[asort,i],label='GP')
          plt.plot(xsort,ytest[asort,i],label='Test')
          plt.ylabel(f'y[{i}]')
          plt.xlabel(f'x[{j}]')
          plt.legend()
          plt.show()

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
      yr = self.target(xr)
      yc = self.yconrevs[0].con(yr)
      return yc

    # Run optimisation
    res = self._core__opt(target_transform,'BO',self.nx,restarts,\
        bounds=bnds,max_iter=max_iter)

    xopt = res.x
    yopt = res.fun

    # Revert to original data
    if revert:
      yopt = self.yconrevs[0].rev(yopt)
      for i in range(self.nx):
        xopt[i] = self.xconrevs[i].rev(xopt[i])

    return xopt,yopt
  """
