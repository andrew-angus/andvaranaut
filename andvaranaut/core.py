#!/bin/python3

import warnings
import cloudpickle as pickle
import numpy as np
from functools import partial
import scipy.stats as st
import ray
import multiprocessing as mp
from time import time as stopwatch
import os
import copy
from scipy.optimize import differential_evolution,NonlinearConstraint,minimize, Bounds
from design import ihs
import pytensor.tensor as pt

# Save and load with pickle
# ToDo: Faster with cpickle 
def save_object(obj,fname):
  with open(fname, 'wb') as f:
    pickle.dump(obj, f)
def load_object(fname):
  with open(fname, 'rb') as f:
    obj = pickle.load(f)
  return obj

# Core class which runs target function
class _core():
  def __init__(self,nx,ny,priors,target,parallel=False,nproc=1,\
      constraints=None,rundir=None,verbose=False):
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
    if not ray.is_initialized():
      ray.init(num_cpus=self.nproc)
    l = len(inps)
    all_ids = [_parallel_wrap.remote(fun,self.rundir,inps[i],i+self.nsamp) for i in range(l)]

    # Get ids as they complete or fail, give warning on fail
    outs = []; fails = np.empty(0,dtype=np.intc)
    id_order = np.empty(0,dtype=np.intc)
    ids = copy.deepcopy(all_ids)
    lold = l; flag = False
    while lold:
      done_id,ids = ray.wait(ids)
      try:
        outs += ray.get(done_id)
        idx = all_ids.index(done_id[0]) 
        id_order = np.append(id_order,idx)
      except:
        idx = all_ids.index(done_id[0]) 
        id_order = np.append(id_order,idx)
        fails = np.append(fails,idx)
        flag = True
        print(f"Warning: parallel run {idx+1} failed with x values {inps[idx]}.",\
          "\nCheck number of inputs/outputs and whether input ranges are valid.")
      lnew = len(ids)
      if lnew != lold:
        lold = lnew
        if self.verbose:
          print(f'Run is {(l-lold)/l:0.1%} complete.',end='\r')
    if flag:
      ray.shutdown()
    
    # Reshape outputs to 2D array
    oldouts = np.array(outs).reshape((len(outs),self.ny))
    outs = np.zeros_like(oldouts)
    outs[id_order] = oldouts

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
    # Parallel execution using ray
    if self.parallel:
      ysamps,fails = self.__parallel_runs(xsamps,fun)
      assert ysamps.shape[1] == self.ny, "Specified ny does not match function output"
    # Serial execution
    else:
      ysamps = np.empty((0,self.ny))
      fails = np.empty(0,dtype=np.intc)
      for i in range(n_samples):
        d = os.path.join(self.rundir, f'task{i+self.nsamp}')
        if not os.path.isdir(d):
          os.system(f'mkdir {d}')
        os.chdir(d)
        # Keep track of fails but run rest of samples
        try:
          yout = fun(xsamps[i,:])
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
        if self.verbose:
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

    # Final print on time taken
    if self.verbose:
      print()
      print(f'Time taken: {t1-t0:0.2f} s')

    return xsamps, ysamps

  # Core optimizer implementing bounds and constraints
  # Global optisation done either with differential evolution or local minimisation with restarts
  def __opt(self,fun,method,nx,restarts=10,nonself=False,priors=None,**kwargs):
    # Construct constraints object if using
    if not nonself:
      if self.constraints is not None:
        cons = self.constraints['constraints']
        upps = self.constraints['upper_bounds']
        lows = self.constraints['lower_bounds']
        nlcs = tuple(NonlinearConstraint(cons[i],lows[i],upps[i]) for i in range(len(cons)))
      else:
        nlcs = tuple()
      kwargs['constraints'] = nlcs

    # Set bounds with priors if exists
    if priors is not None:
      lbs = np.zeros(nx)
      ubs = np.zeros(nx)
      for j in range(nx):
        lbs[j] = priors[j].ppf(1e-8)
        ubs[j] = priors[j].isf(1e-8)
      kwargs['bounds'] = Bounds(lbs,ubs)

    # Global opt method choice
    verbose = self.verbose
    self.verbose = False
    if method == 'DE':
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = differential_evolution(fun,**kwargs)
    elif method == 'restarts':
      # Add buffer to nlcs to stop overshoot
      if not nonself:
        buff = 1e-6
        for i in nlcs:
          i.lb += buff
          i.ub -= buff
      # Draw starting point samples
      points = (ihs(restarts,nx)\
          -1.0+np.random.rand(restarts,nx))/restarts
      # Scale by bounds or priors
      if priors is not None:
        for j in range(nx):
          points[:,j] = priors[j].ppf(points[:,j])
      else:
        bnds = kwargs['bounds']
        points = self.__bounds_scale(points,nx,bnds)
      # Check against constraints and replace if invalid
      if self.constraints is not None and not nonself:
        points = self.__check_constraints(points)
        npoints = len(points)
        # Add points by random sampling and repeat till all valid
        while npoints != restarts:
          nnew = restarts - npoints
          newpoints = np.random.rand(nnew,nx)
          newpoints = self.__bounds_scale(newpoints,nx,bnds)
          newpoints = self.__check_constraints(newpoints)
          points = np.r_[points,newpoints]
          npoints = len(points)

      # Conduct minimisations
      if self.parallel:
        if not ray.is_initialized():
          ray.init(num_cpus=self.nproc)
        # Switch off further parallelism within minimized function
        self.parallel = False
        try:
          results = ray.get([_minimize_wrap.remote(fun,i,**kwargs) for i in points])
          self.parallel = True
        except:
          self.parallel = True
          ray.shutdown()
          raise Exception
      else:
        results = []
        for i in points:
          with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(fun,i,**kwargs)
          results.append(res)

      # Get best result
      f_vals = np.array([i.fun for i in results])
      f_success = [i.success for i in results]
      if not any(f_success):
        print('Warning: All minimizations unsuccesful')
      elif not all(f_success):
        self.verbose = verbose
        if self.verbose:
          print('Removing failed minimizations...')
        f_success = np.where(np.isnan(f_vals),False,f_success)
        f_vals = f_vals[f_success]
        idx = np.arange(len(results))
        idx = idx[f_success]
        results = [results[i] for i in idx]

      best = np.argmin(f_vals)
      res = results[best]


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

  def __bounds_scale(self,points,nx,bnds):
    for i in range(nx):
      points[:,i] *= bnds.ub[i]-bnds.lb[i]
      points[:,i] += bnds.lb[i]
    return points

  # Calculate first derivative with second order central differences
  def __derivative(self,x,fun,idx,eps=1e-6):
    
    # Get shifted input arrays
    xdown = copy.deepcopy(x)
    xup = copy.deepcopy(x)
    xdown[idx] -= eps
    xup[idx] += eps

    # Get function results
    fdown = fun(xdown)
    fup = fun(xup)

    # Calculate derivative
    res = (fup - fdown) / (2*eps)
    return res

  # Calculates gradient vector
  def __grad(self,x,fun,eps=1e-6):
    
    lenx = len(x)
    res = np.zeros(lenx)
    for i in range(lenx):
      res[i] = self.__derivative(x,fun,i,eps)
    return res

  # Calculate hessian matrix
  def __hessian(self,x,fun,eps=1e-6):

    # Compute matrix as jacobian of gradient vector
    lenx = len(x)
    res = np.zeros((lenx,lenx))
    grad = partial(self.__grad,fun=fun,eps=eps)
    for i in range(lenx):
      res[:,i] = self.__derivative(x,grad,i,eps)
    return res 

    # Compute matrix as symmetric derivative of derivative
    lenx = len(x)
    res = np.zeros((lenx,lenx))
    for i in range(lenx):
      for j in range(i):
        div = partial(self.__derivative,fun=fun,eps=eps,idx=i)
        res[i,j] = self.__derivative(x,div,j,eps)
        res[j,i] = res[i,j]
    return res 


# Function which wraps serial function for executing in parallel directories
@ray.remote(max_retries=0)
def _parallel_wrap(fun,rundir,inp,idx):
  d = os.path.join(rundir, f'task{idx}')
  if not os.path.isdir(d):
    os.mkdir(d)
  os.chdir(d)
  res = fun(inp)
  os.chdir('../..')
  return res
 
# Function which wraps serial function for executing in parallel directories
@ray.remote(max_retries=0)
def _minimize_wrap(fun,x0,**kwargs):
  res = minimize(fun,x0,method='SLSQP',**kwargs)
  return res
