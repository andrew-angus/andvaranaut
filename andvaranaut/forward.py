#!/bin/python3

import numpy as np
from design import latin_random
import GPy
import scipy.stats as st
from time import time as stopwatch
from seaborn import kdeplot
import matplotlib.pyplot as plt
from andvaranaut.utils import *

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

  # Private method which takes array of x samples and evaluates y at each
  # ToDo add parallel execution
  def __vector_solver(self,xsamps,parallel,nproc):
    t0 = stopwatch()
    if parallel:
      ysamps = parallel_runs(self.target,xsamps,nproc)
    else:
      n_samples = len(xsamps)
      ysamps = np.zeros((n_samples,self.ny))
      for i in range(n_samples):
        try:
          ysamps[i,:] = self.target(xsamps[i,:])
        except:
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
    print(f'Time taken: {t1-t0:0.1e} s')

    # Add new evaluations to original data arrays
    self.x = np.r_[self.x,xsamps]
    self.y = np.r_[self.y,ysamps]

  # Add n samples to current via latin hypercube sampling
  def sample(self,nsamps,parallel=False,nproc=None):
    points = latin_random(nsamps,self.nx)
    xsamps = np.zeros((nsamps,self.nx))
    for i in range(nsamps):
      for j in range(self.nx):
        xsamps[i,j] = self.dists[j].ppf(points[i,j])
    self.__vector_solver(xsamps,parallel,nproc)

  # Delete n samples by random indexing
  # ToDo: Better to delete by low resolution latin hypercube?
  def del_samples(self,nsamps):
    current = len(self.x)
    left = current-nsamps
    a = np.arange(0,current)
    inds = np.random.choice(a,size=left)
    self.x = self.x[inds,:]
    self.y = self.y[inds,:]

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
