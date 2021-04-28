#!/bin/python3

import numpy as np
from design import latin_random
import GPy

# Latin hypercube sampler and propagator
class lhc():
  def __init__(self,nxvar=None,nyvar=None,dists=None,target=None):
    # Check inputs
    if nxvar is None:
      raise Exception('Error: must specify number of input dimensions')
    # Initialise attributes
    self.nxvar = nxvar # Input dimensions
    self.nyvar = nxvar # Output dimensions
    self.dists = dists # Input distributions (must be scipy)
    self.x = np.empty((0,nxvar))
    self.y = np.empty((0,nyvar))
    # Target function which takes X and returns Y provided by user
    self.target = target

  # Private method which takes array of x samples and evaluates y at each
  # ToDo add parallel execution
  def __vector_solver(self,xsamps):
    t0 = stopwatch()
    n_samples = len(xsamps)
    ysamps = np.zeros((n_samples,1))
    for i in range(n_samples):
      ysamps[i,:] = self.target(xsamps[i,:])
      print(f'Run is {(i+1)/n_samples:0.1%} complete.',end='\r')
    print('Run is 100.0% complete.')
    t1 = stopwatch()
    print(f'Time taken: {t1-t0:0.3f} s')

    # Add new evaluations to original data arrays
    self.x0 = np.r_[self.x0,xsamps]
    self.y0 = np.r_[self.y0,ysamps]


  # Add samples to current via latin hypercube sampling
  def sample(self,nsamps):
    points = latin_random(nsamps,self.nvars)
    xsamps = np.zeros((nsamps,self.nvars))
    for i in range(nsamps):
      for j in range(nvars):
        xsamps[i,j] = self.dists[j].ppf(points[i,j])
    self.__vector_solver(xsamps)

  # Sort samples into ordered form and delete by thinning evenly
  def del_samples(self):
    pass

  # Plot y distribution using kernel density estimation
  def y_dist(self):
    pass

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
