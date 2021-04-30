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

# Latin hypercube sampler and propagator
class lhc():
  def __init__(self,nx=None,ny=None,dists=None,target=None):
    # Check inputs
    if (nx is None) or (nx < 1) or (not isinstance(nx,int)):
      raise Exception('Error: must specify an integer number of input dimensions > 0') 
    if (ny is None) or (ny < 1) or (not isinstance(ny,int)):
      raise Exception('Error: must specify an integer number of output dimensions > 0') 
    if (dists is None) or (len(dists) != nx):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate distributions of length nx') 
    check = 'scipy.stats._distn_infrastructure'
    flags = [not getattr(i,'__module__',None)==check for i in dists]
    if any(flags):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate distributions of length nx') 
    if (target is None) or (not callable(target)):
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
        print(f"Warning: parallel run {i} failed.",\
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
  # ToDo: Better to delete by low resolution latin hypercube?
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
    if self.ny == 1:
      kdeplot(self.y[:,0])
      plt.xlabel('QoI')
      plt.ylabel('Density')
      plt.show()
    else:
      for i in range(self.ny):
        kdeplot(self.y[:,i])
        plt.xlabel(f'y[{i}]')
        plt.ylabel('Density')
        plt.show()

  def scalexy(self):
    # Produce scaled x samples
    x_samps = np.zeros((len_tot,nvars))
    if (nsamps > 0):
      x_samps[:nsamps,:] = newx_samps
    x_samps[nsamps:,:] = old_xsamps
    x_scaled = copy.deepcopy(x_samps)
    for j in range(nvars):
      x_scaled[:,j] = x_convert(x_scaled[:,j],variables[j])

    # Scale y samples
    y_samps = np.zeros((len_tot,1))
    if (nsamps > 0):
      y_samps[:nsamps,:] = newy_samps
    y_samps[nsamps:,:] = old_ysamps
    y_scaled = copy.deepcopy(y_samps)
    y_scaled = y_convert(y_scaled)
 

# Inherit from LHC class and add data conversion methods
class _surrogate(lhc):
  def __init__(self):
    pass

# Inherit from surrogate class and add GP specific methods
class gp(_surrogate):
  def __init__(self):
    self.dat = dat
    self.kern = kernel
    self.nvars = dat.x.shape[1]

  def fit(self,restarts=3,noise=False):

    kernel = 'GPy.kern.'+self.kern+f'(input_dim=)'
    if self.kern == 'RBF':
      kern = self.kern
      #self.model = GPy.models.GPRegression(self.dat.x,self.dat.y,)
