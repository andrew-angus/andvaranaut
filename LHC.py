#!/bin/python3

import numpy as np
from py-design import latin_random

class lhc():
  # Initialise attributes
  def __init__(self,nvars=None,dists=None,target=None):
    self.nvars = nvars
    self.dists = dists
    self.x = np.empty((0,nvars))
    self.y = np.empty((0,1))
    self.x0 = np.empty((0,nvars))
    self.y0 = np.empty((0,1))
    # Target function which takes X and returns Y provided by user
    self.target = target

  # Add samples to current via latin hypercube sampling
  def add_samples(self,nsamps):
    # Get new x and y samples
    points = latin_random(nsamps,self.nvars)
    newx_samps = np.zeros((nsamps,self.nvars))
    for i in range(nsamps):
      for j in range(nvars):
        newx_samps[i,j] = self.dists[j].ppf(points[i,j])
    newy_samps = vector_solver(newx_samps)

    # Get old and new sample combined length
    len_old = len(old_ysamps)
    len_tot = nsamps + len_old

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

  # Sort samples into ordered form and delete by thinning evenly
  def del_samples(self)
    pass

  # Plot y distribution using kernel density estimation
  def y_dist(self):
    pass
