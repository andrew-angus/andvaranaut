#!/bin/python3

import numpy as np
from py-design import latin_random

class lhc():
  # Initialise attributes
  def __init__(self,nvars=None,dists=None,target=None):
    self.nvars = nvars
    self.dists = dists
    # Converted x and y data
    self.x = np.empty((0,nvars))
    self.y = np.empty((0,1))
    # Original x and y data
    self.x0 = np.empty((0,nvars))
    self.y0 = np.empty((0,1))
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
  def del_samples(self)
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
