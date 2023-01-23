#!/bin/python3

import numpy as np
from design import latin_random,ihs
import GPy
from GPyOpt.models import GPModel
from GPyOpt.methods import BayesianOptimization
import scipy.stats as st
from time import time as stopwatch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from functools import partial
from scipy.optimize import minimize,differential_evolution,NonlinearConstraint,Bounds
from andvaranaut.utils import _core,cdf,save_object,wgp
import ray
from matplotlib import ticker
import pymc as pm
import arviz as az

# Latin hypercube sampler and propagator, inherits core
class LHC(_core):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.x = np.empty((0,self.nx))
    self.y = np.empty((0,self.ny))

  # Add n samples to current via latin hypercube sampling
  def sample(self,nsamps,seed=None,improved=True):
    if not isinstance(nsamps,int) or (nsamps < 1):
      raise Exception("Error: nsamps argument must be an integer > 0") 
    if self.verbose:
      print(f'Evaluating {nsamps} latin hypercube samples...')
    xsamps = self.__latin_sample(nsamps,seed,improved)
    if self.constraints is not None:
      xsamps = self._core__check_constraints(xsamps)
    xsamps,ysamps = self._core__vector_solver(xsamps)

    # Add new evaluations to original data arrays
    self.x = np.r_[self.x,xsamps]
    self.y = np.r_[self.y,ysamps]

  # Produce latin hypercube samples from input distributions
  def __latin_sample(self,nsamps,seed=None,improved=True):
    if improved:
      if seed:
        np.random.seed(seed)
      points = (ihs(nsamps,self.nx,seed)\
          -1.0+np.random.rand(nsamps,self.nx))/nsamps
    else:
      points = latin_random(nsamps,self.nx,seed)
    xsamps = np.zeros_like(points)
    for j in range(self.nx):
      xsamps[:,j] = self.priors[j].ppf(points[:,j])
    return xsamps

  # Delete n samples by selected method
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    self.__del_samples(ndels,method,idx,returns=False)

  # Private method with more flexibility for extension in child classes
  def __del_samples(self,ndels,method,idx,returns):
    # Delete samples by proximity to coarse LHC of size (ndels,nx)
    if method == 'coarse_lhc':
      if not isinstance(ndels,int) or ndels < 1:
        raise Exception("Error: must specify positive int for ndels")
      xsamps = self.__latin_sample(ndels)
      dmins = np.zeros(ndels,dtype=np.intc)
      for i in range(ndels):
        lenx = len(self.x)
        dis = np.zeros(lenx)
        for j in range(lenx):
          dis[j] = np.linalg.norm(self.x[j]-xsamps[i])
        dmins[i] = np.argmin(dis)
        self.x = np.delete(self.x,dmins[i],axis=0)
        self.y = np.delete(self.y,dmins[i],axis=0)
      if returns:
        return dmins
    # Delete samples by choosing ndels random indexes
    elif method == 'random':
      if not isinstance(ndels,int) or ndels < 1:
        raise Exception("Error: must specify positive int for ndels")
      current = len(self.x)
      left = current-ndels
      a = np.arange(0,current)
      inds = np.random.choice(a,size=left,replace=False)
      self.x = self.x[inds,:]
      self.y = self.y[inds,:]
      if returns:
        return inds
    # Delete samples at specified indexes
    elif method == 'specific':
      if not isinstance(idx,(int,list)):
        raise Exception("Error: must specify int or list of ints for idx")
      mask = np.ones(len(self.x), dtype=bool)
      mask[idx] = False
      self.x = self.x[mask]
      self.y = self.y[mask]
      if returns:
        return mask
    else:
      raise Exception("Error: method must be one of 'coarse_lhc','random','specific'")

  # Plot y distribution using kernel density estimation and histogram
  def y_dist(self,mode='hist_kde'):
    self.__y_dist(self.y,mode)

  # Private y_dist method with more flexibility for inherited class extension
  def __y_dist(self,y,mode):
    modes = ['hist','kde','ecdf','hist_kde']
    if mode not in modes:
      raise Exception(f"Error: selected mode must be one of {modes}")
    funs = [partial(sns.displot,kind='hist'),partial(sns.displot,kind='kde'),\
            partial(sns.displot,kind='ecdf'),partial(sns.displot,kind='hist',kde=True)]
    for i in range(self.ny):
      funs[modes.index(mode)](y[:,i])
      plt.xlabel(f'y[{i}]')
      plt.ylabel('Density')
      plt.show()

  # Optionally set x and y attributes with existing datasets
  def set_data(self,x,y):
    # Checks that args are 2D numpy float arrays of correct nx/ny
    if not isinstance(x,np.ndarray) or len(x.shape) != 2 \
        or x.dtype != 'float64' or x.shape[1] != self.nx:
      raise Exception(\
          "Error: Setting data requires a 2d numpy array of float64 inputs")
    if not isinstance(y,np.ndarray) or len(y.shape) != 2 \
        or y.dtype != 'float64' or y.shape[1] != self.ny:
      raise Exception(\
          "Error: Setting data requires a 2d numpy array of float64 outputs")
    # Also check if x data within input distribution interval
    for i in range(self.nx):
      intv = self.priors[i].interval(1.0)
      if not all(x[:,i] >= intv[0]) or not all(x[:,i] <= intv[1]):
        raise Exception(\
            "Error: provided x data must fit within provided input distribution ranges.")
    self.x = x
    self.y = y

# Inherit from LHC class and add data conversion methods
class _surrogate(LHC):
  def __init__(self,xconrevs=None,yconrevs=None,**kwargs,):
    # Call LHC init, then validate and set now data conversion/reversion attributes
    super().__init__(**kwargs)
    self.xc = copy.deepcopy(self.x)
    self.yc = copy.deepcopy(self.y)
    self.__conrev_check(xconrevs,yconrevs)
  
  # Update sampling method to include data conversion
  def sample(self,nsamps,seed=None):
    super().sample(nsamps,seed)
    self.__con(len(self.x))

  # Conversion of last n samples
  def __con(self,nsamps):
    self.xc = np.r_[self.xc,np.zeros((nsamps,self.nx))]
    self.yc = np.r_[self.yc,np.zeros((nsamps,self.ny))]
    for i in range(self.nx):
      self.xc[-nsamps:,i] = self.xconrevs[i].con(self.x[-nsamps:,i])
    for i in range(self.ny):
      self.yc[-nsamps:,i] = self.yconrevs[i].con(self.y[-nsamps:,i])

  # Inherit from lhc __del_samples and add converted dataset deletion
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    returned = super()._LHC__del_samples(ndels,method,idx,returns=True)
    if method == 'coarse_lhc':
      for i in range(ndels):
        self.xc = np.delete(self.xc,returned[i],axis=0)
        self.yc = np.delete(self.yc,returned[i],axis=0)
    elif method == 'random':
      self.xc = self.xc[returned,:]
      self.yc = self.yc[returned,:]
    elif method == 'specific':
      self.xc = self.xc[returned]
      self.yc = self.yc[returned]

  # Allow for changing conversion/reversion methods
  def change_conrevs(self,xconrevs=None,yconrevs=None):
    # Check and set new lists, then update converted datasets
    self.__conrev_check(xconrevs,yconrevs)
    for i in range(self.nx):
      self.xc[:,i] = self.xconrevs[i].con(self.x[:,i])
    for i in range(self.ny):
      self.yc[:,i] = self.yconrevs[i].con(self.y[:,i])

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
      self.yc[:,i] = self.yconrevs[i].con(self.y[:,i])

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
    self.__con(len(x))

  # Inherit and extend y_dist to have dist by surrogate predictions
  def y_dist(self,mode='hist_kde',nsamps=None,return_data=False,surrogate=True,predictfun=None):
    # Allow for use of surrogate evaluations or underlying datasets
    if surrogate:
      xsamps = super()._LHC__latin_sample(nsamps)
      xcons = np.zeros((nsamps,self.nx))
      for i in range(self.nx):
        xcons[:,i] = self.xconrevs[i].con(xsamps[:,i])
      ypreds = predictfun(xcons)
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

# Class to replace None types in surrogate conrev arguments
class _none_conrev:
  def con(self,x):
   return x 
  def rev(self,x):
   return x 

# Inherit from surrogate class and add GP specific methods
class GP(_surrogate):
  def __init__(self,kernel='RBF',noise=True,normalise=True,prior_dict=None,**kwargs):
    super().__init__(**kwargs)
    self.change_model(kernel,noise,normalise)
    self.set_prior_dict(prior_dict)
    self.__scrub_train_test()
    self._xRF = None

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
      csamps = len(self.x)
      eseloo = np.zeros((csamps,self.ny))
      for j in range(csamps):

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
    indexes = np.arange(len(self.x))
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
    self.m.kern.lengthscale[:] = hypers[:3]
    self.m.kern.variance = hypers[3]
    if self.noise:
      self.m.Gaussian_noise.variance = hypers[4]

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
  def fit(self,method='mcmc_mean',return_data=False,**kwargs):
    self.m, self.gp, self.hypers, data = self.__fit(self.xc,self.yc,method,**kwargs)
    if return_data:
      return data

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,method,**kwargs):
    
    # PyMC context manager
    m = pm.Model()
    with m:
      # Priors on hyperparameters
      #hdict = self.prior_dict['hypers']
      #kls = [hdict[i].pm for i in range(self.nx)]
      #kvar = hdict[self.nx]
      if self.noise:
        #gvar = hdict[self.nx+1]
        gvar = 1e-8 + pm.HalfNormal('gv',sigma=0.4)
      else:
        gvar = 1e-8
      kls = pm.Gamma('l',alpha=2.15,beta=6.91,shape=3)
      kvar = pm.LogNormal('kv',mu=0.0,sigma=0.4)

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
      y_ = gp.marginal_likelihood("y", X=self.xc, y=self.yc[:,0], noise=gvar)

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
        for j in range(ran):
          if self.xconrevs[i].pos[j]:
            x[rc+j] = np.exp(x[rc+j])
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

# Ray remote function wrap around surrogate prediction
@ray.remote
def _parallel_predict(x,predfun):
  return predfun(x)
