#!/bin/python3

import numpy as np
from design import latin_random
import GPy
import scipy.stats as st
from time import time as stopwatch
import seaborn as sns
import matplotlib.pyplot as plt
import ray
import multiprocessing as mp
import os
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from functools import partial
from scipy.optimize import minimize
from andvaranaut.utils import cdf

# Latin hypercube sampler and propagator
class lhc():
  def __init__(self,nx,ny,dists,target,parallel=False,nproc=1):
    # Check inputs
    if (not isinstance(nx,int)) or (nx < 1):
      raise Exception('Error: must specify an integer number of input dimensions > 0') 
    if (not isinstance(ny,int)) or (ny < 1):
      raise Exception('Error: must specify an integer number of output dimensions > 0') 
    if (not isinstance(dists,list)) or (len(dists) != nx):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate distributions of length nx') 
    check = 'scipy.stats._distn_infrastructure'
    flags = [not getattr(i,'__module__',None)==check for i in dists]
    if any(flags):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate distributions of length nx') 
    if not callable(target):
      raise Exception(\
          'Error: must provide target function which produces output from specified inputs')
    if not isinstance(parallel,bool):
      raise Exception("Error: parallel must be type bool.")
    if not isinstance(nproc,int) or (nproc < 1):
      raise Exception("Error: nproc argument must be an integer > 0")
    assert (nproc <= mp.cpu_count()),\
        "Error: number of processors selected exceeds available."
    # Initialise attributes
    self.nx = nx # Input dimensions
    self.ny = ny # Output dimensions
    self.dists = dists # Input distributions (must be scipy)
    self.x = np.empty((0,nx))
    self.y = np.empty((0,ny))
    self.target = target # Target function which takes X and returns Y
    self.parallel = parallel # Whether to use parallelism wherever possible
    self.nproc = nproc # Number of processors to use if using parallelism

  # Method which takes function, and 2D array of inputs
  # Then runs in parallel for each set of inputs
  # Returning 2D array of outputs
  def __parallel_runs(self,inps):
    # Create parallel directory for tasks
    os.system('mkdir parallel')

    # Run function in parallel in individual directories    
    if not ray.is_initialized():
      ray.init(num_cpus=self.nproc,log_to_driver=False)
    l = len(inps)
    all_ids = [_parallel_wrap.remote(self.target,inps[i],i) for i in range(l)]

    # Get ids as they complete or fail, give warning on fail
    outs = []; fails = np.empty(0,dtype=np.intc)
    ids = copy.deepcopy(all_ids)
    lold = l
    while lold:
      done_id,ids = ray.wait(ids)
      try:
        outs += ray.get(done_id)
      except:
        idx = all_ids.index(done_id[0]) 
        fails = np.append(fails,idx)
        print(f"Warning: parallel run {idx+1} failed with samples {inps[idx]}.",\
          "\nCheck number of inputs/outputs and whether input ranges are valid.")
      lnew = len(ids)
      if lnew != lold:
        lold = lnew
        print(f'Run is {(l-lold)/l:0.1%} complete.',end='\r')
    #ray.shutdown()
    
    # Reshape outputs to 2D array
    outs = np.array(outs).reshape((len(outs),self.ny))

    return outs, fails

  # Private method which takes array of x samples and evaluates y at each
  def __vector_solver(self,xsamps):
    t0 = stopwatch()
    n_samples = len(xsamps)
    # Parallel execution using ray
    if self.parallel:
      ysamps,fails = self.__parallel_runs(xsamps)
      assert ysamps.shape[1] == self.ny, "Specified ny does not match function output"
    # Serial execution
    else:
      ysamps = np.empty((0,self.ny))
      fails = np.empty(0,dtype=np.intc)
      for i in range(n_samples):
        # Keep track of fails but run rest of samples
        try:
          yout = self.target(xsamps[i,:])
        except:
          errstr = f"Warning: Target function evaluation failed at sample {i+1} "+\
              "with xsamples: " +str(xsamps[i,:])+\
              "\nCheck number of inputs and range of input values valid."
          print(errstr)
          fails = np.append(fails,i)
          continue
        # Number of function outputs check
        try:
          ysamps = np.vstack((ysamps,yout))
        except:
          raise Exception("Error: number of target function outputs is not equal to ny")
        print(f'Run is {(i+1)/n_samples:0.1%} complete.',end='\r')
    print()
    t1 = stopwatch()
    print(f'Time taken: {t1-t0:0.2f} s')

    # Remove failed samples
    mask = np.ones(n_samples, dtype=bool)
    mask[fails] = False
    xsamps = xsamps[mask]

    # Add new evaluations to original data arrays
    self.x = np.r_[self.x,xsamps]
    self.y = np.r_[self.y,ysamps]

  # Add n samples to current via latin hypercube sampling
  def sample(self,nsamps):
    if not isinstance(nsamps,int) or (nsamps < 1):
      raise Exception("Error: nsamps argument must be an integer > 0")
    print(f'Evaluating {nsamps} latin hypercube samples...')
    xsamps = self.__latin_sample(nsamps)
    self.__vector_solver(xsamps)

  # Produce latin hypercube samples from input distributions
  def __latin_sample(self,nsamps):
    points = latin_random(nsamps,self.nx)
    xsamps = np.zeros_like(points)
    for j in range(self.nx):
      xsamps[:,j] = self.dists[j].ppf(points[:,j])
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
    # Checks that args are 2D numpy float arrays
    if not isinstance(x,np.ndarray) or len(x.shape) != 2 or x.dtype != 'float64':
      raise Exception(\
          "Error: Setting data requires a 2d numpy array of float64 inputs")
    if not isinstance(y,np.ndarray) or len(y.shape) != 2 or y.dtype != 'float64':
      raise Exception(\
          "Error: Setting data requires a 2d numpy array of float64 outputs")
    # Also check if x data within input distribution interval
    for i in range(self.nx):
      intv = self.dists[i].interval(1.0)
      if not all(x[:,i] >= intv[0]) or not all(x[:,i] <= intv[1]):
        raise Exception(\
            "Error: provided x data must fit within provided input distribution ranges.")
    self.x = x
    self.y = y

# Function which wraps serial function for executing in parallel directories
@ray.remote(max_retries=0)
def _parallel_wrap(fun,inp,idx):
  d = f'./parallel/task{idx}'
  os.system(f'mkdir {d}')
  os.chdir(d)
  res = fun(inp)
  os.chdir('../..')
  return res
 
# Inherit from LHC class and add data conversion methods
class _surrogate(lhc):
  def __init__(self,xconrevs=None,yconrevs=None,**kwargs,):
    # Call LHC init, then validate and set now data conversion/reversion attributes
    super().__init__(**kwargs)
    self.xc = copy.deepcopy(self.x)
    self.yc = copy.deepcopy(self.y)
    self.__conrev_check(xconrevs,yconrevs)
  
  # Update sampling method to include data conversion
  def sample(self,nsamps):
    super().sample(nsamps)
    self.__con(nsamps)

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
    returned = super()._lhc__del_samples(ndels,method,idx,returns=True)
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
      xsamps = super()._lhc__latin_sample(nsamps)
      xcons = np.zeros((nsamps,self.nx))
      for i in range(self.nx):
        xcons[:,i] = self.xconrevs[i].con(xsamps[:,i])
      ypreds = predictfun(xcons)
      yrevs = np.zeros((nsamps,self.ny))
      for i in range(self.ny):
        yrevs[:,i] = self.yconrevs[i].rev(ypreds[:,i])
      amax = np.argmax(ypreds)
      idx = (amax//self.ny,amax%self.ny)
      super()._lhc__y_dist(yrevs,mode)
      if return_data:
        return xsamps,yrevs
    elif not surrogate:
      super().y_dist(mode)
    else:
      raise Exception("Error: surrogate argument must be of type bool")

# Class to replace None types in surrogate conrev arguments
class _none_conrev:
  def __init__(self):
    pass
  def con(self,x):
   return x 
  def rev(self,x):
   return x 

# Inherit from surrogate class and add GP specific methods
class gp(_surrogate):
  def __init__(self,kernel='RBF',noise=True,**kwargs):
    super().__init__(**kwargs)
    self.change_model(kernel,noise)
    self.__scrub_train_test()
    self._xRF = None

  def __scrub_train_test(self):
    self.xtrain = None
    self.xtest = None
    self.ytrain = None
    self.ytest = None

  # Sample method inherits lhc sampling and adds adaptive sampling option
  def sample(self,nsamps,method='lhc',batchsize=1,restarts=3):
    methods = ['lhc','adaptive']
    if method not in methods:
      raise Exception(f'Error: method must be one of {methods}')
    if method == 'lhc':
      super().sample(nsamps)
    else:
      if self.x.shape[0] < 2:
        raise Exception("Error: require at least 2 LHC samples to perform adaptive sampling.")
      self.__adaptive_sample(nsamps,batchsize,restarts)

  # Adaptive sampler based on Mohammadi, Hossein, et al. 
  # "Cross-validation based adaptive sampling for Gaussian process models." 
  # arXiv preprint arXiv:2005.01814 (2020).
  def __adaptive_sample(self,nsamps,batchsize,restarts):
    # Determine number of batches and new samples in each batch
    batches = round(np.ceil(nsamps/batchsize))
    for i in range(batches):
      newsamps = np.minimum(nsamps-i*batches,batchsize)

      # Fit GP to current samples
      self.fit(restarts)

      # Keep hyperparameters and evaluate ESE_loo for each data point
      csamps = len(self.x)
      eseloo = np.zeros((csamps,self.ny))
      print('Calculating ESE_loo...')
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
          ym = pr[0,k]; yv = prvar[0,k]
          fxi = self.yc[j,k]
          yse = np.power(ym-fxi,2)
          Eloo = yv + yse
          Varloo = 2*yv**2 + 4*yv*yse
          eseloo[j,k] = Eloo/np.sqrt(Varloo)
        print(f'{(j+1)/csamps:0.1%} complete.',end='\r')
      print()
      
      # Fit auxillary GP to ESE_loo
      ## Better to initiate separate gp class instance here?
      print('Fitting auxillary GP to ESE_loo data...')
      gpaux = gp(kernel='RBF',noise=False,nx=self.nx,ny=self.ny,\
                 xconrevs=[cdf(self.dists[i]) for i in range(self.nx)],yconrevs=None,\
                 parallel=self.parallel,\
                 nproc=self.nproc,dists=self.dists,\
                 target=self.target)
      gpaux.set_data(self.x,eseloo)
      gpaux.fit(restarts=restarts)

      # Check lengthscales are not too small
      # Min lengthscale estimated using current x dataset
      maxrscaled = gpaux._gp__K_of_r_root()
      maxr = self.__max_distance()
      minl = maxr/maxrscaled
      for i in range(self.nx):
        if gpaux.m.kern.lengthscale[i] < minl:
          gpaux.m.kern.lengthscale[i] = minl
      print(gpaux.m[''])
      gpaux.test_plots(opt=False)

      # Create repulsive function dataset
      gpaux._gp__create_xRF()

      # Get sample batch
      for j in range(newsamps):
        pass
        # Maximise PEI to get next sample then add to RF data

      # Evaluate function at sample points and add to database

  # Function for finding r which gives Kroot
  def __K_of_r_root(self,Kroot=1e-8):
    def min_fun(r,Kroot):
      return np.abs(self.m.kern.K_of_r(r)-Kroot)
    return minimize(min_fun,1.0,args=Kroot,tol=Kroot/100).x[0]

  # Function for approximating max distance between points in dataset
  def __max_distance(self):
    xmaxs = np.array([np.max(self.xc[:,i]) for i in range(self.nx)])
    xmins = np.array([np.min(self.xc[:,i]) for i in range(self.nx)])
    return np.linalg.norm(xmaxs-xmins)

  # Create repulsive function dataset
  def __create_xRF(self):
    # Existing dataset
    self._xRF = copy.deepcopy(self.xc)
    # Add corners of input space
    xmaxs = np.array([self.xconrevs[i].con(self.dists[i].ppf(1))\
        for i in range(self.nx)])
    xmins = np.array([self.xconrevs[i].con(self.dists[i].isf(1))\
        for i in range(self.nx)])
    xmaxmins = ([xmaxs[i],xmins[i]] for i in range(self.nx))
    xgrid = np.meshgrid(*xmaxmins)
    xcoords = np.array(list(zip(*(i.flat for i in xgrid))))
    self._xRF = np.r_[self._xRF,xcoords]
    # Add closest boundary points to current dataset
    amaxs = np.argmax(self.xc,axis=0)
    amins = np.argmin(self.xc,axis=0)
    xmaxds = self.xc[amaxs]
    xminds = self.xc[amins]
    for i in range(self.nx):
      xmaxds[i,i] = xmaxs[i]
      xminds[i,i] = xmins[i]
    self._xRF = np.r_[self._xRF,xmaxds,xminds]

  # RF function
  def __RF(self,x):
    prod = 1.0
    for i in self._xRF:
      i = np.expand_dims(i,axis=0)
      prod *= 1-self.m.kern.K(x,i)[0,0]/self.m.kern.variance[0]
      #prod += np.log(1-ExpKern(x,i))
    return prod

  # Expected improvement function
  def __EI(x):
    maxye = np.max(eseloo)
    devthresh = 1e-4
    stdnorm = st.norm()
    pr,prvar = maux.predict(np.array([x]))
    ydev = np.sqrt(prvar[0,0])
    if ydev < devthresh:
      return 0
    else:
      ydiff = pr[0,0]-maxye
      u = ydiff/ydev
      return ydiff*stdnorm.cdf(u)+ydev*stdnorm.pdf(u)

  # Pseudo-expected improvement func
  def __PEI(self,x):
  #   return -(np.log(EI(x))+np.log(RF(x)))
    return EI(x)*RF(x)

  # Inherit del_samples and extend to remove test-train datasets
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    super().del_samples(ndels,method,idx)
    self.__scrub_train_test()

  # Fit GP standard method
  def fit(self,restarts=3):
    self.m = self.__fit(self.xc,self.yc,restarts=restarts)

  # More flexible private fit method which can use unconverted or train-test datasets
  def __fit(self,x,y,restarts=3,opt=True):
    # Get correct GPy kernel
    kstring = 'GPy.kern.'+self.kernel+'(input_dim=self.nx,variance=1.,lengthscale=1.,ARD=True)'
    kern = eval(kstring)
    # Fit and return model
    if not self.noise:
      meps = np.finfo(np.float64).eps
      m = GPy.models.GPRegression(x,y,kern,noise_var=meps,normalizer=True)
      m.likelihood.fix()
    else:
      m = GPy.models.GPRegression(x,y,kern,noise_var=1.0,normalizer=True)
    if opt:
      t0 = stopwatch()
      print('Optimizing hyperparameters...')
      m.optimize_restarts(restarts,parallel=self.parallel,num_processes=self.nproc,robust=True)
      t1 = stopwatch()
      print(f'Time taken: {t1-t0:0.2f} s')
    return m

  # Standard predict method which wraps the GPy predict and allows for parallelism
  def predict(self,x,return_var=False):
    return self.__predict(self.m,x,return_var)

  # Private predict method with more flexibility to act on any provided GPy model
  def __predict(self,m,x,return_var=False):
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
      ypreds = np.empty((0,self.ny)); yvarpreds = np.empty((0,self.ny))
      for i in range(chunks):
        ypreds = np.r_[ypreds,outs[i][0]]
        yvarpreds = np.r_[yvarpreds,outs[i][1]]
      #ray.shutdown()
    else:
      ypreds,yvarpreds = m.predict(x)
    t1 = stopwatch()
    print(f'Time taken: {t1-t0:0.2f} s')
    if return_var:
      return ypreds,yvarpreds
    else:
      return ypreds

  # Make train-test split and populate attributes
  def train_test(self,training_frac=0.9):
    self.xtrain,self.xtest,self.ytrain,self.ytest = \
      train_test_split(self.xc,self.yc,train_size=training_frac)

  # Assess GP performance with several test plots and RMSE calcs
  def test_plots(self,restarts=3,revert=True,yplots=True,xplots=True,opt=True):
    # Creat train-test sets if none exist
    if self.xtrain is None:
      self.train_test()
    # Train model on training set and make predictions on xtest data
    mtrain = self.__fit(self.xtrain,self.ytrain,restarts=restarts,opt=opt)
    if not opt and self.m is not None:
      mtrain.kern.lengthscale = self.m.kern.lengthscale
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
    print()
    for i in range(self.ny):
      rmse = np.sqrt(np.sum((ypred[:,i]-ytest[:,i])**2)/len(ytest[:,i]))
      print(f'RMSE for y[{i}] is: {rmse}')
    # Compare ytest and predictions for each output variable
    if yplots:
      for i in range(self.ny):
        plt.title(f'y[{i}]')
        plt.plot(ytest[:,i],ytest[:,i],'-',label='True')
        plt.plot(ytest[:,i],ypred[:,i],'x',label='GP')
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
  def change_model(self,kernel,noise):
    kerns = ['RBF','Matern52','Matern32','Exponential']
    if kernel not in kerns:
      raise Exception(f"Error: kernel must be one of {kerns}")
    if not isinstance(noise,bool):
      raise Exception(f"Error: noise must be of type bool")
    self.kernel = kernel
    self.noise = noise
    self.m = None

  # Inherit set_data method and scrub train-test sets
  def set_data(self,x,y):
    super().set_data(x,y)
    self.__scrub_train_test()

  # Inherit and extend y_dist to have dist by surrogate predictions
  def y_dist(self,mode='hist_kde',nsamps=None,return_data=False,surrogate=True):
    return super().y_dist(mode,nsamps,return_data,surrogate,self.predict)

  # Function which plots relative importances (inverse lengthscales)
  def relative_importances(self,original_data=False,restarts=3):
    if original_data:
      m = self.__fit(self.x,self.y,restarts=restarts)
    else:
      m = self.m
    sens_gp = np.zeros(self.nx)
    for i in range(self.nx):
      leng = m.kern.lengthscale[i]
      sens_gp[i] = self.dists[i].mean()/leng
    plt.bar([f'x[{i}]'for i in range(self.nx)],np.log(sens_gp))
    plt.ylabel('Relative log importance')
    plt.show()

# Ray remote function wrap around surrogate prediction
@ray.remote
def _parallel_predict(x,predfun):
  return predfun(x)
