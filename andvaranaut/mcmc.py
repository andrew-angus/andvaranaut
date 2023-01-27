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
from andvaranaut.utils import _core,cdf,save_object,wgp
from andvaranaut.forward import _surrogate
import ray
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# Inherit from surrogate class and add GP specific methods
class GPyMC(_surrogate):
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
      #if self.x.shape[0] < 2:
        #  raise Exception("Error: require at least 2 LHC samples to perform adaptive sampling.")
      #self.__adaptive_sample(nsamps,batchsize,restarts,opt_method,opt_restarts)

  # Inherit del_samples and extend to remove test-train datasets
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    super().del_samples(ndels,method,idx)
    self.__scrub_train_test()

  # Fit GP standard method
  def fit(self,method='mcmc_mean',return_data=False,iwgp=False,cwgp=False,**kwargs):
    self.m, self.gp, self.hypers, data = self.__fit(self.xc,self.yc,method,\
        iwgp,cwgp,**kwargs)
    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,method,iwgp,cwgp,**kwargs):
    
    # PyMC context manager
    m = pm.Model()
    with m:
      # Priors on hyperparameters
      #hdict = self.prior_dict['hypers']
      #kls = [hdict[i].pm for i in range(self.nx)]
      #kvar = hdict[self.nx]
      if self.noise:
        #gvar = hdict[self.nx+1]
        gvar = pm.HalfNormal('gv',sigma=0.4)
      else:
        gvar = 1e-8
      kls = pm.Gamma('l',alpha=2.15,beta=6.91,shape=3)
      #kvar = pm.LogNormal('kv',mu=0.0,sigma=0.4)
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
      if cwgp:
        y_ = gp.marginal_likelihood("y", X=xin, y=self.yc[:,0], noise=gvar) \
            + 1.0
      else:
        y_ = gp.marginal_likelihood("y", X=xin, y=self.yc[:,0], noise=gvar)

      # Fit
      if method == 'map':
        data = pm.find_MAP(**kwargs)
        mp = copy.deepcopy(data)
      else:
        data = pm.sample(**kwargs)
        if method == 'mcmc_mean':
          mean = data.posterior.mean(dim=["chain", "draw"])
          mp = {}
          for key in mean:
            if len(mean[key].values.shape) > 1:
              mp[key] = mean[key].values
            else:
              mp[key] = np.array(mean[key].values)
        else:
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

    return m, gp, mp, data

  # Set output warping parameters
  def cwgp_set(self,x):
    for i in range(len(x)):
      if self.yconrevs[0].pos[i]:
        x[i] = np.exp(x[i])
    self.change_yconrevs(yconrevs=[wgp(self.yconrevs[0].warping_names,x,self.y)])

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
    #self.xtrain,self.xtest,self.ytrain,self.ytest = \
    #  train_test_split(self.xc,self.yc,train_size=training_frac)
    indexes = np.arange(len(self.x))
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

  def model_likelihood(self):
    # Base log likelihood from GPy
    baseLL = self.m.log_likelihood()

    # Warping term
    warp = 0
    for i in range(self.ny):
      warp += np.sum(np.log(self.yconrevs[i].der(self.y)))

    return baseLL + warp

  # Model posterior with prior distribution dict passed as argument
  def model_posterior(self):
    LL = self.model_likelihood()
    pri = 0
    if 'hypers' in self.prior_dict:
      for i,j in enumerate(self.prior_dict['hypers']):
        if j is not None:
          if i < 3:
            pri += j.logpdf(np.log(self.m.kern.lengthscale[i]))
          elif i == 3:
            pri += j.logpdf(np.log(self.m.kern.variance[0]))
          else:
            pri += j.logpdf(np.log(self.m.Gaussian_noise.variance[0]))
    if 'cwgp' in self.prior_dict:
      for i,j in enumerate(self.prior_dict['cwgp']):
        if self.yconrevs[0].pos[i]:
          pri += j.logpdf(np.log(self.yconrevs[0].params[i]))
        else:
          pri += j.logpdf(self.yconrevs[0].params[i])

    return LL + pri

  # Method for optimising parameters
  def param_opt(self,method='restarts',restarts=10,opt_hypers=True,\
      opt_cwgp=False,opt_iwgp=False,posterior=True,**kwargs):

    # Establish number of parameters and populate default priors if necessary
    nx = 0
    priors = []
    splits = []
    if opt_hypers:
      priors.extend(self.prior_dict['hypers'])
      if self.noise:
        num = 5
      else:
        num = 4
      nx += num
      splits.append(nx)
    if opt_cwgp:
      if 'cwgp' not in self.prior_dict:
        self.prior_dict['cwgp'] = self.yconrevs[0].default_priors
      npar = len(self.yconrevs[0].params) 
      priors.extend(self.prior_dict['cwgp'])
      nx += npar
      splits.append(nx)
    if opt_iwgp:
      if 'iwgp' not in self.prior_dict:
        self.prior_dict['iwgp'] = []
        npar = 0
        for i in range(self.nx):
          if isinstance(self.xconrevs[i],wgp):
            self.prior_dict['iwgp'].extend(self.xconrevs[i].default_priors)
            npar += 2
      else:
        npar = len(self.prior_dict['iwgp'])
      priors.extend(self.prior_dict['iwgp'])
      nx += npar
      splits.append(nx)

    if self.verbose:
      print(f'Optimising {nx} parameters...')

    # Setup objective function
    if opt_iwgp and opt_cwgp and opt_hypers:
      if posterior:
        obj_fun = partial(self.__param_opt_phic,splits=splits)
      else:
        obj_fun = partial(self.__param_opt_lhic,splits=splits)
    elif opt_cwgp and opt_hypers:
      if posterior:
        obj_fun = partial(self.__param_opt_phc,splits=splits)
      else:
        obj_fun = partial(self.__param_opt_lhc,splits=splits)
    elif opt_iwgp and opt_hypers:
      if posterior:
        obj_fun = partial(self.__param_opt_phi,splits=splits)
      else:
        obj_fun = partial(self.__param_opt_lhi,splits=splits)
    elif opt_iwgp and opt_cwgp:
      if posterior:
        obj_fun = partial(self.__param_opt_pic,splits=splits)
      else:
        obj_fun = partial(self.__param_opt_lic,splits=splits)
    elif opt_cwgp:
      if posterior:
        obj_fun = self.__param_opt_pc
      else:
        obj_fun = self.__param_opt_lc
    elif opt_hypers:
      if posterior:
        obj_fun = self.__param_opt_ph
      else:
        obj_fun = self.__param_opt_lh
    elif opt_iwgp:
      if posterior:
        obj_fun = self.__param_opt_pi
      else:
        obj_fun = self.__param_opt_li

    # Call optimisation method
    res = self._core__opt(obj_fun,method,nx,\
        nonself=True,priors=priors,restarts=restarts,**kwargs)

    # Ensure opt used
    obj_fun(res.x)

  def __param_opt_lhc(self,x,splits):
    self.hyper_set(x[:splits[0]])
    self.cwgp_set(x[splits[0]:])
    return -self.model_likelihood()

  def __param_opt_phc(self,x,splits):
    self.hyper_set(x[:splits[0]])
    self.cwgp_set(x[splits[0]:])
    return -self.model_posterior()

  def __param_opt_lh(self,x):
    self.hyper_set(x)
    return -self.model_likelihood()

  def __param_opt_ph(self,x):
    self.hyper_set(x)
    return -self.model_posterior()

  def __param_opt_lc(self,x):
    self.cwgp_set(x)
    self.fit(2)
    return -self.model_likelihood()

  def __param_opt_pc(self,x):
    self.cwgp_set(x)
    self.fit(2)
    return -self.model_posterior()

  def __param_opt_li(self,x):
    self.iwgp_set(x)
    self.fit(2)
    return -self.model_likelihood()

  def __param_opt_pi(self,x):
    self.iwgp_set(x)
    self.fit(2)
    return -self.model_posterior()

  def __param_opt_lic(self,x,splits):
    self.cwgp_set(x[:splits[0]])
    self.iwgp_set(x[splits[0]:])
    self.fit(2)
    return -self.model_likelihood()

  def __param_opt_pic(self,x,splits):
    self.cwgp_set(x[:splits[0]])
    self.iwgp_set(x[splits[0]:])
    self.fit(2)
    return -self.model_posterior()

  def __param_opt_lhi(self,x,splits):
    self.hyper_set(x[:splits[0]])
    self.iwgp_set(x[splits[0]:])
    return -self.model_likelihood()

  def __param_opt_phi(self,x,splits):
    self.hyper_set(x[:splits[0]])
    self.iwgp_set(x[splits[0]:])
    return -self.model_posterior()

  def __param_opt_lhic(self,x,splits):
    self.hyper_set(x[:splits[0]])
    self.cwgp_set(x[splits[0]:splits[1]])
    self.iwgp_set(x[splits[1]:])
    return -self.model_likelihood()

  def __param_opt_phic(self,x,splits):
    self.hyper_set(x[:splits[0]])
    self.cwgp_set(x[splits[0]:splits[1]])
    self.iwgp_set(x[splits[1]:])
    return -self.model_posterior()

  def hyper_set(self,x):
    hypers = np.exp(x)
    self.m.kern.lengthscale[:] = hypers[:3]
    self.m.kern.variance = hypers[3]
    if self.noise:
      self.m.Gaussian_noise.variance = hypers[4]
    
  # Modified conversion class change method
  def change_conrevs(self,xconrevs=None,yconrevs=None):
    super().change_conrevs(xconrevs=xconrevs,yconrevs=yconrevs)
    if self.m is not None:
      self.m.set_XY(self.xc,self.yc)

  # Modified conversion class change method
  def change_xconrevs(self,xconrevs=None):
    super().change_xconrevs(xconrevs=xconrevs)
    if self.m is not None:
      self.m.set_X(self.xc)

  # Modified conversion class change method
  def change_yconrevs(self,yconrevs=None):
    super().change_yconrevs(yconrevs=yconrevs)
    if self.m is not None:
      self.m.set_Y(self.yc)

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
