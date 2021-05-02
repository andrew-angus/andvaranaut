#!/bin/python3

import numpy as np
from design import latin_random
import GPy
import scipy.stats as st
from time import time as stopwatch
from seaborn import kdeplot
import matplotlib.pyplot as plt
import ray
import multiprocessing as mp
import os
import numpy as np
import copy

# Latin hypercube sampler and propagator
class lhc():
  def __init__(self,nx,ny,dists,target):
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
    # Initialise attributes
    self.nx = nx # Input dimensions
    self.ny = ny # Output dimensions
    self.dists = dists # Input distributions (must be scipy)
    self.x = np.empty((0,nx))
    self.y = np.empty((0,ny))
    # Target function which takes X and returns Y provided by user
    self.target = target

  # Function which wraps serial function for executing in parallel directories
  @ray.remote
  def __parallel_wrap(self,inp,idx):
    d = f'./parallel/task{idx}'
    os.system(f'mkdir {d}')
    os.chdir(d)
    res = self.target(inp)
    os.chdir('../..')
    return res

  # Method which takes function, and 2D array of inputs
  # Then runs in parallel for each set of inputs
  # Returning 2D array of outputs
  def __parallel_runs(self,inps,nps):
      
    # Ensure number of requested processors is reasonable
    assert (nps <= mp.cpu_count()),\
        "Error: number of processors selected exceeds available."
    
    # Create parallel directory for tasks
    os.system('mkdir parallel')

    # Run function in parallel    
    ray.init(num_cpus=nps,log_to_driver=False,ignore_reinit_error=True)
    l = len(inps)
    ids = []
    for i in range(len(inps)):
      ids += [self.__parallel_wrap.remote(self,inps[i],i)]
    outs = []; fail = -1
    for i in range(len(inps)):
      try:
        outs.append(ray.get(ids[i]))
      except:
        fail = i
        print(f"Warning: parallel run {i+1} failed with samples {inps[i]}.",\
          "Check number of inputs/outputs and whether input ranges are valid.",\
          "Will save previous successful runs to database.")
        ray.shutdown()
        break
    ray.shutdown()
    
    # Reshape outputs to 2D array
    if isinstance(outs[0],np.ndarray):
      outs = np.array(outs)
    else:
      outs = np.array(outs).reshape((l,1))

    return outs, fail

  # Private method which takes array of x samples and evaluates y at each
  def __vector_solver(self,xsamps,parallel,nproc):
    t0 = stopwatch()
    if parallel:
      # Parallel execution using ray
      ysamps,fail = self.__parallel_runs(xsamps,nproc)
      assert ysamps.shape[1] == self.ny, "Specified ny does not match function output"
      if fail > -1:
        xsamps = xsamps[:fail]
    else:
      # Serial execution
      n_samples = len(xsamps)
      ysamps = np.zeros((n_samples,self.ny))
      for i in range(n_samples):
        try:
          yout = self.target(xsamps[i,:])
          if isinstance(yout,np.ndarray):
            assert len(yout) == self.ny
          else:
            assert self.ny == 1
          ysamps[i,:] = yout
        except:
          # If a sample evaluation fails still dump succesful samples
          self.x = np.r_[self.x,xsamps[:i,:]]
          self.y = np.r_[self.y,ysamps[:i,:]]
          errstr = f"Error: Target function evaluation failed at sample {i+1} "+\
              "with xsamples: " +str(xsamps[i,:])+\
              "\nCheck number of inputs/outputs is correct and range of input values valid."+\
              f"\n{i} samples succesfully added to dataset."
          raise Exception(errstr)
        print(f'Run is {(i+1)/n_samples:0.1%} complete.',end='\r')
    print('Run is 100.0% complete.')
    t1 = stopwatch()
    print(f'Time taken: {t1-t0:0.2f} s')

    # Add new evaluations to original data arrays
    self.x = np.r_[self.x,xsamps]
    self.y = np.r_[self.y,ysamps]

  # Add n samples to current via latin hypercube sampling
  def sample(self,nsamps,parallel=False,nproc=None):
    points = latin_random(nsamps,self.nx)
    xsamps = np.array([[self.dists[j].ppf(points[i,j]) \
              for j in range(self.nx)]for i in range(nsamps)])
    self.__vector_solver(xsamps,parallel,nproc)

  # Delete n samples by random indexing
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    if method == 'coarse_lhc':
      if not isinstance(ndels,int) or ndels < 1:
        raise Exception("Error: must specify positive int for ndels")
      points = latin_random(ndels,self.nx)
      xsamps = np.array([[self.dists[j].ppf(points[i,j]) \
                for j in range(self.nx)]for i in range(ndels)])
      for i in range(ndels):
        lenx = len(self.x)
        dis = np.zeros(lenx)
        for j in range(lenx):
          dis[j] = np.linalg.norm(self.x[j]-xsamps[i])
        dmin = np.argmin(dis)
        self.x = np.delete(self.x,dmin,axis=0)
        self.y = np.delete(self.y,dmin,axis=0)
    elif method == 'random':
      if not isinstance(ndels,int) or ndels < 1:
        raise Exception("Error: must specify positive int for ndels")
      current = len(self.x)
      left = current-ndels
      a = np.arange(0,current)
      inds = np.random.choice(a,size=left,replace=False)
      self.x = self.x[inds,:]
      self.y = self.y[inds,:]
    elif method == 'specific':
      if not isinstance(idx,(int,list)):
        raise Exception("Error: must specify int or list of ints for idx")
      mask = np.ones(len(self.x), dtype=bool)
      mask[idx] = False
      self.x = self.x[mask]
      self.y = self.y[mask]
    else:
      raise Exception("Error: method must be one of 'coarse_lhc','random','specific'")

  # Plot y distribution using kernel density estimation
  def y_dist(self):
    for i in range(self.ny):
      kdeplot(self.y[:,i])
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
 
# Inherit from LHC class and add data conversion methods
class _surrogate(lhc):
  def __init__(self,nx,ny,dists,target,\
                xconrevs=None,yconrevs=None):
    # Call LHC init and add converted dataset attributes
    super().__init__(nx,ny,dists,target)
    self.xc = copy.deepcopy(self.x)
    self.yc = copy.deepcopy(self.y)
    # Validate provided data conversion & reversion functions
    flag = False
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
        if not flag:
          flag = True
          print("Warning: One or more data conversion/reversion method is None.",\
              "This may affect surrogate performance.")
    self.xconrevs = xconrevs
    self.yconrevs = yconrevs
  
  # Update sampling method to include data conversion
  def sample(self,nsamps,parallel=False,nproc=None):
    super().sample(nsamps,parallel,nproc)
    self.__con(nsamps)

  # Conversion of last n samples
  def __con(self,nsamps):
    self.xc = np.r_[self.xc,np.zeros((nsamps,self.nx))]
    self.yc = np.r_[self.yc,np.zeros((nsamps,self.ny))]
    for i in range(self.nx):
      self.xc[-nsamps:,i] = self.xconrevs[i].con(self.x[-nsamps:,i])
    for i in range(self.ny):
      self.yc[-nsamps:,i] = self.yconrevs[i].con(self.y[-nsamps:,i])

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
  def __init__(self,nx,ny,dists,target,\
                xconrevs=None,yconrevs=None):
    super().__init__(nx,ny,dists,target,xconrevs,yconrevs)

  def fit(self):
    pass
