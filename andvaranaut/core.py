#!/bin/python3
import warnings
import cloudpickle as pickle
import numpy as np
from functools import partial
import scipy.stats as st
import dask
from dask.distributed import Client
import multiprocessing as mp
from time import time as stopwatch
import os
import copy
from scipy.optimize import differential_evolution,NonlinearConstraint,minimize, Bounds
from scipy.stats import qmc
import pytensor.tensor as pt
from netCDF4 import *
from time import sleep
from tqdm import trange

# Save and load with cloudpickle
def save_object(obj,fname):
  with open(fname, 'wb') as f:
    pickle.dump(obj, f)
def load_object(fname):
  with open(fname, 'rb') as f:
    obj = pickle.load(f)
  return obj

# Save and load plot data netCDF
def save_xy(x,y=None,fname='savexy.nc'):
  f = Dataset(fname,'w')
  n = f.createDimension('n',len(x))
  xdat = f.createVariable('x','f8',('n'))
  if y is not None:
    ydat = f.createVariable('y','f8',('n'))
    ydat[:] = y
  xdat[:] = x
  f.close()

# Load netCDF saved xy-data
def load_xy(fname,xonly=False):
  f = Dataset(fname,'r')
  x = f.variables['x'][:]
  if not xonly:
    y = f.variables['y'][:]
  f.close()
  if not xonly:
    return x,y
  else:
    return x

# Core class which runs target function
class _core():
  def __init__(self,nx,ny,priors,target,parallel=False,nproc=1,\
      constraints=None,rundir=None,verbose=True,pulse=1):

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
    if (not isinstance(constraints,dict)) and (constraints is not None):
      raise Exception(\
          f'Error: provided constraints must be a dictionary with keys {keys} and list items.') 
    keys = ['constraints','lower_bounds','upper_bounds']
    if constraints is not None:
      if not all(key in constraints for key in keys):
        raise Exception(\
          f'Error: provided constraints must be a dictionary with keys {keys} and list items.') 
    # Initialise attributes
    self.nx = nx # Input dimensions
    self.ny = ny # Output dimensions
    self.priors = priors # Input distributions (must be scipy)
    self.target = target # Target function which takes X and returns Y
    self.parallel = parallel # Whether to use parallelism wherever possible
    self.nproc = nproc # Number of processors to use if using parallelism
    self.pulse = pulse # Seconds to check parallel task completion
    self.constraints = constraints # List of constraint functions for sampler
    self.verbose = verbose
    self.rundir = 'runs'
    self.nsamp = 0
    if rundir is not None:
      self.rundir = rundir

  # Method which takes function, and 2D array of inputs
  # Then runs in parallel for each set of inputs
  # Returning 2D array of outputs
  def __parallel_runs(self,inps,fun):

    # Run function in parallel in individual directories    
    # with dask client
    with Client(n_workers=self.nproc,threads_per_worker=1) as client:

      # Start tasks and initialise output arrays
      l = len(inps)
      futures = [client.submit(_parallel_wrap,fun=fun,rundir=self.rundir, \
          inp=inps[i],idx=i+self.nsamp) for i in range(l)]
      outs = np.empty((0,self.ny))
      fails = np.empty(0,dtype=np.intc)
      
      # Check for completed tasks at pulse interval and handle errors
      unfinished = np.arange(l)
      while len(unfinished) > 0:
        sleep(self.pulse)
        for i in unfinished:
          status = futures[i].status
          # Save results of finished tasks
          if status == 'finished':
            result = futures[i].result()
            outs = np.r_[outs,np.array([result])]
            unfinished = np.delete(unfinished,np.where(unfinished == i))
          # Save sample id of failed tasks
          elif status == 'error':
            fails = np.append(fails,i)
            unfinished = np.delete(unfinished,np.where(unfinished == i))

    return outs, fails

  # Private method which takes array of x samples and evaluates y at each
  def __vector_solver(self,xsamps,fun=None):
    if fun is None:
      fun = self.target
    t0 = stopwatch()
    n_samples = len(xsamps)

    # Create directory for tasks
    if not os.path.isdir(self.rundir):
      os.mkdir(self.rundir)

    # Parallel execution using dask
    if self.parallel:
      ysamps,fails = self.__parallel_runs(xsamps,fun)
      assert ysamps.shape[1] == self.ny, "Specified ny does not match function output"
      for i in fails:
        errstr = f"Warning: Target function evaluation failed at sample {i} "+\
            "with x values: " +str(xsamps[i,:]) 
        print(errstr)

    # Serial execution
    else:
      ysamps = np.empty((0,self.ny))
      fails = np.empty(0,dtype=np.intc)
      # Progress bar if verbose
      if self.verbose:
        rangef = trange
      else:
        rangef = range
      for i in rangef(n_samples):
        d = os.path.join(self.rundir, f'task{i+self.nsamp}')
        if not os.path.isdir(d):
          os.system(f'mkdir {d}')
        os.chdir(d)
        # Keep track of fails but run rest of samples
        try:
          yout = fun(xsamps[i,:])
        except Exception as e:
          errstr = f"Warning: Target function evaluation failed at sample {i} "+\
              "with x values: " +str(xsamps[i,:]) + "; error message: " 
          errstr = errstr + str(e)
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

    # Final print on time taken
    if self.verbose:
      print()
      print(f'Time taken: {t1-t0:0.2f} s')

    return xsamps, ysamps

  # Check proposed samples against all provided constraints
  def __check_constraints(self,xsamps):
    nsamps0 = len(xsamps)
    mask = np.ones(nsamps0,dtype=bool)
    for i,j in enumerate(xsamps):
      for e,f in enumerate(self.constraints['constraints']):
        flag = True
        res = f(j)
        lower_bounds = self.constraints['lower_bounds'][e]
        upper_bounds = self.constraints['upper_bounds'][e]
        if isinstance(lower_bounds,list):
          for k,l in enumerate(lower_bounds):
            if res[k] < l:
              flag = False
          for k,l in enumerate(upper_bounds):
            if res[k] > l:
              flag = False
        else:
          if res < lower_bounds:
            flag = False
          elif res > upper_bounds:
            flag = False
        mask[i] = flag
        if not flag:
          print(f'Sample {i+1} with x values {j} removed due to invalidaing constraint {e+1}.')
    xsamps = xsamps[mask]
    nsamps1 = len(xsamps)
    if nsamps1 < nsamps0:
      print(f'{nsamps0-nsamps1} samples removed due to violating constraints.')
    return xsamps

# Function which wraps serial function for executing in parallel directories
def _parallel_wrap(fun,rundir,inp,idx):
  d = os.path.join(rundir, f'task{idx}')
  if not os.path.isdir(d):
    os.mkdir(d)
  os.chdir(d)
  res = fun(inp)
  os.chdir('../..')
  return res
