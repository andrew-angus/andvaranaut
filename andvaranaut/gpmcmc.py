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
from scipy.optimize import Bounds
from scipy.stats._continuous_distns import uniform_gen, norm_gen

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
  def fit(self,method='map',return_data=False,\
      iwgp=False,cwgp=False,**kwargs):
    self.m, self.gp, self.hypers, data = self.__fit(self.x,self.y,\
        method,iwgp,cwgp,**kwargs)

    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,method,iwgp,cwgp,**kwargs):
    
    # PyMC context manager
    m = pm.Model()
    nsamp = len(y)
    with m:
      # Priors on GP hyperparameters
      if self.noise:
        gvar = pm.HalfNormal('gv',sigma=0.4)
      else:
        gvar = 0.0
      kls = pm.LogNormal('l',mu=0.0,sigma=1.0,shape=self.nx)
      kvar = pm.LogNormal('kv',mu=0.56,sigma=0.75)

      # Input warping
      if iwgp:
        rc = 0
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            rc += self.xconrevs[i].np
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
          cwgpp = pm.TruncatedNormal('cwgp_pos',mu=1.0,sigma=1.0,\
              lower=1e-3,upper=5.0,shape=rcpos)
        if rc > 0:
          cwgp = pm.TruncatedNormal('cwgp',mu=0.0,sigma=1.0,\
              lower=-5,upper=5,shape=rc)
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
      if cwgp:
        K = kern(xin)
        K += pt.identity_like(K)*(1e-6+gvar)
        L = pt.slinalg.cholesky(K)
        beta = pt.slinalg.solve_triangular(L,yin,lower=True)
        alpha = pt.slinalg.solve_triangular(L.T,beta)
        y_ = pm.Potential('ypot',-0.5*pt.dot(yin.T,alpha)\
            -pt.sum(pt.log(pt.diag(L)))\
            -0.5*nsamp*pt.log(2*np.pi)\
            +pt.sum(pt.log(yder)))
      else:
        gp = pm.gp.Marginal(cov_func=kern)
        y_ = gp.marginal_likelihood("y", X=xin, y=yin, noise=pt.sqrt(gvar))

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
          mp = pm.find_MAP(start=mp)
        else:
          raise Exception('method must be one of map, mcmc_map, or mcmc_mean')

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
      y = self.y
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
    
    y = self.__predict(self.m,self.gp,self.hypers,xarg,return_var)

    if revert:
      for i in range(self.ny):
        y[0][:,i] = self.yconrevs[i].rev(y[0][:,i])

    if self.verbose:
      print("warning: reversion of variance not implemented")

    return y

  # Private predict method with more flexibility to act on any provided GPy model
  def __predict(self,m,gp,hyps,x,return_var=False):
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
        ypreds, yvarpreds = gp.predict(x, point=hyps,  diag=True)
    t1 = stopwatch()
    if self.verbose:
      print(f'Time taken: {t1-t0:0.2f} s')
    if return_var:
      return ypreds.reshape((-1,1)),yvarpreds.reshape((-1,1))
    else:
      return ypreds.reshape((-1,1))

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
      yc = self.yconrevs[0].con(yr[:,0])
      # Update database with new BO samples
      self.x = np.r_[self.x,xr]
      self.y = np.r_[self.y,yr]
      self.xc = np.r_[self.xc,np.array([xc])]
      self.yc = np.r_[self.yc,np.array([yc])]
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
        yopt = self.yconrevs[0].rev(yopt)
        for i in range(self.nx):
          xopt[i] = self.xconrevs[i].rev(xopt[i])

    except:
      print('Warning: Bayesian minimisation failed, choosing best sample in dataset')
      yopt = np.min(self.y)
      xopt = self.x[np.argmin(self.y[:,0]),:]


    self.xopt = xopt
    self.yopt = yopt

    return xopt,yopt

  # Fit GP with inverse optimisation
  def inverse_opt(self,yobs,\
      method='mcmc_mean',return_data=False,iwgp=False,**kwargs):
    self.m, self.gp, self.hypers, data = self.__inverse(yobs,method,\
        iwgp,**kwargs)
    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __inverse(self,yobs,method,iwgp,**kwargs):
    
    from andvaranaut.transform import kumaraswamy
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

  # Assess GP performance with several test plots and RMSE calcs
  def test_plots(self,revert=True,yplots=True,xplots=True\
      ,iwgp=False,cwgp=False,method='none'):
    # Creat train-test sets if none exist
    if self.train is None:
      self.train_test()
    self.xtrain = self.x[self.train,:]
    self.xtest = self.x[self.test,:]
    self.ytrain = self.y[self.train,:]
    self.ytest = self.y[self.test,:]

    # Train model on training set and make predictions on xtest data
    m, gp, hypers, data = self.__fit(self.xtrain,self.ytrain,method,iwgp,cwgp)
    if self.verbose:
      print(hypers)
    xtest = np.zeros_like(self.xtest)
    for i in range(self.nx):
      xtest[:,i] = self.xconrevs[i].con(self.xtest[:,i])
    ypred = self.__predict(m,gp,hypers,xtest,return_var=False)

    # Either revert data to original for comparison or leave as is
    if revert:
      xtest = self.xtest
      ytest = self.ytest
      ypred = self.yconrevs[0].rev(ypred[:,0])
    else: 
      ytest = self.yconrevs[0].con(self.ytest[:,0])

    # RMSE for each y variable
    rmse = np.sqrt(np.sum((ypred-ytest)**2)/len(ytest))
    if self.verbose:
      print(f'RMSE for y is: {rmse:0.5e}')
    # Compare ytest and predictions for each output variable
    if yplots:
      plt.title(f'y')
      plt.plot(ytest,ytest,'-',label='True')
      plt.plot(ytest,ypred,'X',label='GP')
      plt.ylabel(f'y')
      plt.xlabel(f'y')
      plt.legend()
      plt.show()
    # Compare ytest and predictions wrt each input variable
    if xplots:
      for j in range(self.nx):
        plt.title(f'y wrt x[{j}]')
        xsort = np.sort(xtest[:,j])
        asort = np.argsort(xtest[:,j])
        plt.plot(xsort,ypred[asort],label='GP')
        plt.plot(xsort,ytest[asort],label='Test')
        plt.ylabel(f'y')
        plt.xlabel(f'x[{j}]')
        plt.legend()
        plt.show()

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
  """
