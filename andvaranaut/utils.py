#!/bin/python3

import pickle
from scipy.special import logit as lg
import numpy as np
from functools import partial
import scipy.stats as st
import ray
import multiprocessing as mp
from time import time as stopwatch
import os
import copy

# Save and load with pickle
# ToDo: Faster with cpickle 
def save_object(obj,fname):
  with open(fname, 'wb') as f:
    pickle.dump(obj, f)
def load_object(fname):
  with open(fname, 'rb') as f:
    obj = pickle.load(f)
  return obj

## Conversion functions

# Vectorised logit, with bounds checking to stop inf
@np.vectorize
def __logit(x):
  bnd = 0.9999999999999999
  x = np.minimum(bnd,x)
  x = np.maximum(1.0-bnd,x)
  return lg(x)
# Logit transform using uniform dists
def logit(x,dist):
  # Convert uniform distribution samples to standard uniform [0,1]
  # and logit transform to unbounded range 
  x01 = cdf_con(x,dist)
  x = __logit(x01)
  return x
# Convert uniform dist samples into standard uniform 
def std_uniform(x,dist):
  intv = dist.interval(1.0)
  x = (x-intv[0])/(intv[1]-intv[0])
  return x
# Convert normal dist samples into standard normal
def std_normal(x,dist):
  x = (x-dist.mean())/dist.std()
  return x
# Convert positive values to unbounded with logarithm
def log_con(y):
  return np.log(y)
# Convert non-negative to unbounded, via intermediate [0,1]
def nonneg_con(y):
  y01 = y/(1+y)
  return  __logit(y01)
# Probit transform to standard normal using scipy dist
def probit_con(x,dist):
  std_norm = st.norm()
  xcdf = np.where(x<0,1-dist.sf(x),dist.cdf(x))
  x = np.where(xcdf<0.5,std_norm.isf(1-xcdf),std_norm.ppf(xcdf))
  return x
# Transform any dist to standard uniform using cdf
def cdf_con(x,dist):
  x = np.where(x<dist.mean(),1-dist.sf(x),dist.cdf(x))
  return x
# Normalise by provided factor
def normalise_con(y,fac):
  return y/fac

## Reversion functions

# Vectorised logistic, with bounds checking to stop inf and
# automatic avoidance of numbers close to zero for numerical accuracy
@np.vectorize
def __logistic(x):
  bnd = 36.73680056967710072513000341132283210754394531250
  sign = np.sign(x)
  x = np.minimum(bnd,x)
  x = np.maximum(-bnd,x)
  ex = np.exp(sign*x)
  return 0.50-sign*0.50+sign*ex/(ex+1.0)
# Logistic transform using uniform dists
def logistic(x,dist):
  x01 = __logistic(x)
  x = cdf_rev(x01,dist)
  return x
# Revert to original uniform distributions
def uniform_rev(x,dist):
  intv = dist.interval(1.0)
  x = x*(intv[1]-intv[0])+intv[0]
  return x
# Revert to original uniform distributions
def normal_rev(x,dist):
  x = x*dist.std()+dist.mean()
  return x
# Revert logarithm with power
def log_rev(y):
  #return np.power(10,y)
  return np.exp(y)
# Revert unbounded to non-negative, via intermediate [0,1]
def nonneg_rev(y):
  y01 = __logistic(y)
  return  y01/(1-y01)
# Reverse probit transform from standard normal using scipy dist
def probit_rev(x,dist):
  std_norm = st.norm()
  xcdf = np.where(x<0,1-std_norm.sf(x),std_norm.cdf(x))
  x = np.where(xcdf<0.5,dist.isf(1-xcdf),dist.ppf(xcdf))
  return x
# Transform any dist to standard uniform using cdf
def cdf_rev(x,dist):
  x = np.where(x<0.5,dist.isf(1-x),dist.ppf(x))
  return x
# Revert standard normalisation
def normalise_rev(y,fac):
  return y*fac

# Define class wrappers for matching sets of conversions and reversions
# Also allows a standard format for use in surrogates without worrying about function arguments
class normal:
  def __init__(self,dist):
    self.con = partial(std_normal,dist=dist)
    self.rev = partial(normal_rev,dist=dist)
class uniform:
  def __init__(self,dist):
    self.con = partial(std_uniform,dist=dist)
    self.rev = partial(uniform_rev,dist=dist)
class logit_logistic:
  def __init__(self,dist):
    self.con = partial(logit,dist=dist)
    self.rev = partial(logistic,dist=dist)
class probit:
  def __init__(self,dist):
    self.con = partial(probit_con,dist=dist)
    self.rev = partial(probit_rev,dist=dist)
class cdf:
  def __init__(self,dist):
    self.con = partial(cdf_con,dist=dist)
    self.rev = partial(cdf_rev,dist=dist)
class nonneg:
  def __init__(self):
    self.con = nonneg_con
    self.rev = nonneg_rev
class logarithm:
  def __init__(self):
    self.con = log_con
    self.rev = log_rev
class normalise:
  def __init__(self,fac):
    self.con = partial(normalise_con,fac=fac)
    self.rev = partial(normalise_rev,fac=fac)

# Core class which runs target function
class _core():
  def __init__(self,nx,ny,priors,target,parallel=False,nproc=1):
    # Check inputs
    if (not isinstance(nx,int)) or (nx < 1):
      raise Exception('Error: must specify an integer number of input dimensions > 0') 
    if (not isinstance(ny,int)) or (ny < 1):
      raise Exception('Error: must specify an integer number of output dimensions > 0') 
    if (not isinstance(priors,list)) or (len(priors) != nx):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate priors of length nx') 
    check = 'scipy.stats._distn_infrastructure'
    flags = [not getattr(i,'__module__',None)==check for i in priors]
    if any(flags):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate priors of length nx') 
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
    self.priors = priors # Input distributions (must be scipy)
    self.target = target # Target function which takes X and returns Y
    self.parallel = parallel # Whether to use parallelism wherever possible
    self.nproc = nproc # Number of processors to use if using parallelism

  # Method which takes function, and 2D array of inputs
  # Then runs in parallel for each set of inputs
  # Returning 2D array of outputs
  def __parallel_runs(self,inps,verbose):

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
        print(f"Warning: parallel run {idx+1} failed with x values {inps[idx]}.",\
          "\nCheck number of inputs/outputs and whether input ranges are valid.")
      lnew = len(ids)
      if lnew != lold:
        lold = lnew
        if verbose:
          print(f'Run is {(l-lold)/l:0.1%} complete.',end='\r')
    #ray.shutdown()
    
    # Reshape outputs to 2D array
    outs = np.array(outs).reshape((len(outs),self.ny))

    return outs, fails

  # Private method which takes array of x samples and evaluates y at each
  def __vector_solver(self,xsamps,verbose=True):
    t0 = stopwatch()
    n_samples = len(xsamps)
    # Create directory for tasks
    os.system('mkdir runs')
    # Parallel execution using ray
    if self.parallel:
      ysamps,fails = self.__parallel_runs(xsamps,verbose)
      assert ysamps.shape[1] == self.ny, "Specified ny does not match function output"
    # Serial execution
    else:
      ysamps = np.empty((0,self.ny))
      fails = np.empty(0,dtype=np.intc)
      for i in range(n_samples):
        d = f'./runs/task{i}'
        os.system(f'mkdir {d}')
        os.chdir(d)
        # Keep track of fails but run rest of samples
        try:
          yout = self.target(xsamps[i,:])
        except:
          errstr = f"Warning: Target function evaluation failed at sample {i+1} "+\
              "with x values: " +str(xsamps[i,:])+\
              "\nCheck number of inputs and range of input values valid."
          print(errstr)
          fails = np.append(fails,i)
          os.chdir('../..')
          continue


        # Number of function outputs check and append samples
        try:
          ysamps = np.vstack((ysamps,yout))
        except:
          os.chdir('../..')
          raise Exception("Error: number of target function outputs is not equal to ny")
        os.chdir('../..')
        if verbose:
          print(f'Run is {(i+1)/n_samples:0.1%} complete.',end='\r')
    t1 = stopwatch()

    # Remove failed samples
    mask = np.ones(n_samples, dtype=bool)
    mask[fails] = False
    xsamps = xsamps[mask]

    # NaN and inf check
    fails = np.empty(0,dtype=np.intc)
    for i,j in enumerate(ysamps):
      if np.any(np.isnan(j)) or np.any(np.abs(j) == np.inf):
        fails = np.append(fails,i)
        errstr = f"Warning: Target function evaluation returned inf/nan at sample "+\
            "with x values: " +str(xsamps[i,:])+"\nCheck range of input values valid."
        print(errstr)
    mask = np.ones(len(xsamps),dtype=bool)
    mask[fails] = False
    xsamps = xsamps[mask]
    ysamps = ysamps[mask]

    # Final print on successful samples 
    if verbose:
      print()
      print(f'Sampling complete with {len(xsamps)/n_samples:0.1%} success rate.')
      print(f'Time taken: {t1-t0:0.2f} s')

    return xsamps, ysamps

# Function which wraps serial function for executing in parallel directories
@ray.remote(max_retries=0)
def _parallel_wrap(fun,inp,idx):
  d = f'./runs/task{idx}'
  os.system(f'mkdir {d}')
  os.chdir(d)
  res = fun(inp)
  os.chdir('../..')
  return res
 
