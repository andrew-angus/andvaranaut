#!/bin/python3

import numpy as np
import GPy
from GPyOpt.models import GPModel
from GPyOpt.methods import BayesianOptimization
import scipy.stats as st
from time import time as stopwatch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy
from sklearn.model_selection import train_test_split
from functools import partial
from scipy.optimize import minimize,differential_evolution,NonlinearConstraint,Bounds
from andvaranaut.core import _core,save_object
from andvaranaut.transform import cdf,wgp
from andvaranaut.lhc import _surrogate, _none_conrev
import ray
from matplotlib import ticker

# Inherit from surrogate class and add GP specific methods
class GP(_surrogate):
  def __init__(self,kernel='RBF',noise=True,normalise=True,prior_dict=None,**kwargs):
    super().__init__(**kwargs)
    self.change_model(kernel,noise,normalise)
    self.set_prior_dict(prior_dict)
    self.__scrub_train_test()
    self._xRF = None
    self.train = None
    self.test = None

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

  # Sample method inherits lhc sampling and adds adaptive sampling option
  def sample(self,nsamps,method='lhc',batchsize=1,restarts=10,seed=None,opt_method='DE',opt_restarts=10):
    methods = ['lhc','adaptive']
    if method not in methods:
      raise Exception(f'Error: method must be one of {methods}')
    if method == 'lhc':
      super().sample(nsamps,seed)
    else:
      if self.x.shape[0] < 2:
        raise Exception("Error: require at least 2 LHC samples to perform adaptive sampling.")
      self.__adaptive_sample(nsamps,batchsize,restarts,opt_method,opt_restarts)

  # Adaptive sampler based on Mohammadi, Hossein, et al. 
  # "Cross-validation based adaptive sampling for Gaussian process models." 
  # arXiv preprint arXiv:2005.01814 (2020).
  def __adaptive_sample(self,nsamps,batchsize,restarts,opt_method,opt_restarts):
    # Determine number of batches and new samples in each batch
    batches = round(float(np.ceil(nsamps/batchsize)))
    if self.verbose:
      print(f'Sampling {batches} batches of size {batchsize}...')
    t0 = stopwatch()
    for i in range(batches):
      newsamps = np.minimum(nsamps-i*batchsize,batchsize)

      # Fit GP to current samples
      verbose = self.verbose
      self.verbose = False
      self.fit(restarts)

      # Keep hyperparameters and evaluate ESE_loo for each data point
      eseloo = np.zeros((self.nsamp,self.ny))
      for j in range(self.nsamp):

        # Delete single sample
        x_loo = np.delete(self.xc,j,axis=0)
        y_loo = np.delete(self.yc,j,axis=0)

        # Fit GP using saved hyperparams
        mloo = self.__fit(x_loo,y_loo,restarts=restarts,opt=False)
        mloo.kern.variance = self.m.kern.variance
        mloo.kern.lengthscale = self.m.kern.lengthscale
        if self.noise:
          mloo.Gaussian_noise.variance = self.m.Gaussian_noise.variance

        # Predict mean and variance at deleted sample point
        pr,prvar = mloo.predict(self.xc[j:j+1])

        # Calculate ESE_loo
        ##Todo: Revert here before calculating ese_loo?
        for k in range(self.ny):
          ym = pr[0,k]
          #yv = prvar[0,k]
          yv = np.mean(prvar)
          #ym = self.yconrevs[k].rev(pr[0,k])
          #yv = self.yconrevs[k].rev(np.sqrt(prvar[0,k]))**2
          fxi = self.yc[j,k]
          #fxi = self.y[j,k]
          yse = np.power(ym-fxi,2)
          Eloo = yv + yse
          Varloo = 2*yv**2 + 4*yv*yse
          eseloo[j,k] = Eloo/np.sqrt(Varloo)
      
      # Fit auxillary GP to ESE_loo
      xconrevs = [cdf(self.priors[j]) for j in range(self.nx)]
      yconrevs = [_none_conrev() for j in range(self.ny)]
      gpaux = GP(kernel='RBF',noise=False,nx=self.nx,ny=self.ny,\
                 xconrevs=xconrevs,yconrevs=yconrevs,\
                 normalise=False,parallel=self.parallel,\
                 nproc=self.nproc,priors=self.priors,\
                 target=self.target,constraints=copy.deepcopy(self.constraints))
      gpaux.set_data(self.x,eseloo-np.sqrt(0.5))
      gpaux.m = gpaux._GP__fit(gpaux.xc,gpaux.yc,restarts=restarts,minl=True)
      #if self.verbose:
        #print(gpaux.m[''])

      # Create repulsive function dataset
      gpaux._xRF = copy.deepcopy(gpaux.xc)

      # Get sample batch
      xsamps = np.zeros((newsamps,self.nx))
      # Set bounds and constraints
      bnd = 1e-10
      bnds = Bounds([bnd for j in range(self.nx)],[1-bnd for j in range(self.nx)])
      #if self.verbose:
        #print(bnds)
      kwargs = {'bounds':bnds}
      if self.constraints is not None:
        gpaux.constraints['constraints'] = [partial(gpaux._GP__cconstraint,constraint=j) \
            for j in self.constraints['constraints']]
      if opt_method =='DE':
        kwargs['polish'] = False
      self.verbose = verbose
      for j in range(newsamps):
        # Maximise PEI to get next sample then add to RF data
        res = gpaux._core__opt(gpaux._GP__negative_PEI,opt_method,gpaux.nx,opt_restarts,**kwargs)
        x1 = np.expand_dims(res.x,axis=0)
        # Add multiple copies of this to batch to avoid clustering
        for k in range(10):
          gpaux._xRF = np.r_[gpaux._xRF,x1]
        xsamps[j] = res.x
      
      self.verbose = verbose
      #plt.scatter(gpaux.xc[:,0],gpaux.xc[:,1])
      #plt.scatter(xsamps[:,0],xsamps[:,1])
      #plt.xlim(0,1)
      #plt.ylim(0,1)
      #plt.show()

      # Evaluate function at sample points and add to database
      for j in range(self.nx):
        xsamps[:,j] = gpaux.xconrevs[j].rev(xsamps[:,j])
      self.verbose=False
      xsamps,ysamps = self._core__vector_solver(xsamps)
      self.verbose = verbose
      self.x = np.r_[self.x,xsamps]
      self.y = np.r_[self.y,ysamps]
      self.nsamp = len(self.x)
      self._surrogate__con(newsamps)
      if self.verbose:
        print('Sample:',xsamps)
        print(f'{(i+1)/batches:0.1%} complete.',end='\r')
      
    t1 = stopwatch()
    if self.verbose:
      print(f'Time taken: {t1-t0:0.2f} s')

  # Wrapper of constraint function which operates on converted inputs for adaptive sampling opt
  def __cconstraint(self,xc,constraint):
    x = np.zeros_like(xc)
    for i in range(self.nx):
      x[i] = self.xconrevs[i].rev(xc[i])
    return constraint(x)

  # Function for finding r which gives Kroot
  def __K_of_r_root(self,Kroot=1e-10):
    def min_fun(r,Kroot):
      return np.abs(self.m.kern.K_of_r(r)-Kroot)
    return minimize(min_fun,1.0,args=Kroot,tol=Kroot/100).x[0]

  # RF function
  def __RF(self,x):
    x = np.expand_dims(x,axis=0)
    prod = 1.0
    #prod = 0.0
    #tiny = np.finfo(np.float64).tiny
    for i in self._xRF:
      i = np.expand_dims(i,axis=0)
      prod *= 1-self.m.kern.K(x,i)[0,0]/self.m.kern.variance[0]
      #prod += np.log(np.maximum(1-self.m.kern.K(x,i)[0,0]/self.m.kern.variance[0],tiny))
    return prod

  # Expected improvement function
  # If multiple outputs average EI is returned
  def __EI(self,x):
    x = np.expand_dims(x,axis=0)
    maxye = np.max(self.yc)
    stdnorm = st.norm()
    pr,prvar = self.m.predict(x)
    EIs = np.zeros(self.ny)
    tiny = np.finfo(np.float64).tiny
    for i in range(self.ny):
      ydev = np.sqrt(np.mean(prvar))
      if ydev == 0:
        EIs[i] = 0
      else:
        ydiff = pr[0,i]-maxye
        u = ydiff/ydev
        EIs[i] = ydiff*stdnorm.cdf(u)+ydev*stdnorm.pdf(u)
        #EIs[i] = np.exp(np.log(np.maximum(ydiff,tiny))+stdnorm.logcdf(u))\
        #    +np.exp(np.log(np.maximum(ydev,tiny))+stdnorm.logpdf(u))
    #return np.log(np.maximum(np.mean(EIs),tiny))
    return np.mean(EIs)

  # Pseudo-expected improvement func
  def __PEI(self,x):
    res = np.log(np.maximum(self.__EI(x),np.finfo(np.float64).tiny))\
        +np.log(np.maximum(self.__RF(x),np.finfo(np.float64).tiny))
    #res = self.__EI(x)*self.__RF(x)
    #res = self.__EI(x) + self.__RF(x)
    return res
        
  # Negative PEI for minimisation
  def __negative_PEI(self,x):
    #print(f'Iteration: {self.iter}; x: {x}; constraint_fun: {self.constraints["constraints"][0](x)}')
    #self.iter += 1
    return -self.__PEI(x)

  # Inherit del_samples and extend to remove test-train datasets
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    super().del_samples(ndels,method,idx)
    self.__scrub_train_test()

  # Fit GP standard method
  def fit(self,restarts=10):
    self.m = self.__fit(self.xc,self.yc,restarts)

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,restarts=10,opt=True,minl=False):
    # Get correct GPy kernel
    kstring = 'GPy.kern.'+self.kernel+'(input_dim=self.nx,variance=1.,lengthscale=1.,ARD=True)'
    kern = eval(kstring)
    # Fit and return model
    if not self.noise:
      #meps = np.finfo(np.float64).eps
      m = GPy.models.GPRegression(x,y,kern,noise_var=1e-8,normalizer=self.normalise)
      m.likelihood.fix()
    else:
      m = GPy.models.GPRegression(x,y,kern,noise_var=1.0,normalizer=self.normalise)
    if opt:
      t0 = stopwatch()
      if minl:
        self.m = m
        maxrscaled = self.__K_of_r_root()
        minl = 1.0/maxrscaled
        m.kern.lengthscale.constrain_bounded(minl,1e6,warning=False)
      if self.verbose:
        print('Optimizing hyperparameters...')
      m.optimize_restarts(restarts,parallel=self.parallel,\
          num_processes=self.nproc,robust=True,verbose=self.verbose)
      t1 = stopwatch()
      if self.verbose:
        print(f'Time taken: {t1-t0:0.2f} s')
    return m

  # Standard predict method which wraps the GPy predict and allows for parallelism
  def predict(self,x,return_var=False,convert=False,revert=False):
    if convert:
      xarg = np.zeros_like(x)
      for i in range(self.nx):
        xarg[:,i] = self.xconrevs[i].con(x[:,i])
    else:
      xarg = copy.deepcopy(x)
    
    y = self.__predict(self.m,xarg,return_var)

    if revert and not return_var:
      for i in range(self.ny):
        y[:,i] = self.yconrevs[i].rev(y[:,i])
    elif revert:
      raise Exception("Reversion of variance not implemented, set return_var = False")

    return y

  # Private predict method with more flexibility to act on any provided GPy model
  def __predict(self,m,x,return_var=False):
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
      outs = ray.get([_parallel_predict.remote(\
          x[cinds[i]:cinds[i+1]].reshape((csizes[i],self.nx)),\
          m.predict) for i in range(chunks)]) # Chunks
      ypreds = np.empty((0,self.ny)); yvarpreds = np.empty((0,outs[0][1].shape[1]))
      for i in range(chunks):
        ypreds = np.r_[ypreds,outs[i][0]]
        yvarpreds = np.r_[yvarpreds,outs[i][1]]
      #ray.shutdown()
    else:
      ypreds,yvarpreds = m.predict(x)
    t1 = stopwatch()
    if self.verbose:
      print(f'Time taken: {t1-t0:0.2f} s')
    if return_var:
      return ypreds,yvarpreds
    else:
      return ypreds

  # Make train-test split and populate attributes
  def train_test(self,training_frac=0.9):
    #self.xtrain,self.xtest,self.ytrain,self.ytest = \
    #  train_test_split(self.xc,self.yc,train_size=training_frac)
    indexes = np.arange(self.nsamp)
    self.train,self.test = \
      train_test_split(indexes,train_size=training_frac)

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

  # Method to change noise/kernel attributes, scrubs any saved model
  def change_model(self,kernel,noise,normalise):
    kerns = ['RBF','Matern52','Matern32','Exponential']
    if kernel not in kerns:
      raise Exception(f"Error: kernel must be one of {kerns}")
    if not isinstance(noise,bool):
      raise Exception(f"Error: noise must be of type bool")
    self.kernel = kernel
    self.noise = noise
    self.normalise = normalise
    self.m = None

  # Inherit set_data method and scrub train-test sets
  def set_data(self,x,y):
    super().set_data(x,y)
    self.__scrub_train_test()

  # Inherit and extend y_dist to have dist by surrogate predictions
  def y_dist(self,mode='hist_kde',nsamps=None,return_data=False,surrogate=True):
    return super().y_dist(mode,nsamps,return_data,surrogate,self.predict)

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
    try:
      for i in range(self.ny):
        warp += np.sum(np.log(self.yconrevs[i].der(self.y)))
    except:
      pass

    return baseLL + warp

  # Model posterior with prior distribution dict passed as argument
  def model_posterior(self):
    LL = self.model_likelihood()
    pri = 0
    if 'hypers' in self.prior_dict:
      for i,j in enumerate(self.prior_dict['hypers']):
        if j is not None:
          if i < self.nx:
            pri += j.logpdf(np.log(self.m.kern.lengthscale[i]))
          elif i == self.nx:
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

    # Check m exists
    if self.m is None:
      self.m = self.__fit(self.xc,self.yc,opt=False)
    # Establish number of parameters and populate default priors if necessary
    nx = 0
    priors = []
    splits = []
    if opt_hypers:
      priors.extend(self.prior_dict['hypers'])
      if self.noise:
        num = self.nx + 2
      else:
        num = self.nx + 1
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
    self.m.kern.lengthscale[:] = hypers[:self.nx]
    self.m.kern.variance = hypers[self.nx]
    if self.noise:
      self.m.Gaussian_noise.variance = hypers[self.nx+1]

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
        for j in range(ran):
          if self.xconrevs[i].pos[j]:
            x[rc+j] = np.exp(x[rc+j])
        xconrevs.append(wgp(self.xconrevs[i].warping_names,x[rc:rc+ran]\
            ,y=self.x[:,i],xdist=self.priors[i]))
        rc += ran
      else:
        xconrevs.append(self.xconrevs[i])
    self.change_xconrevs(xconrevs=xconrevs)
    
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


# Ray remote function wrap around surrogate prediction
@ray.remote
def _parallel_predict(x,predfun):
  return predfun(x)
