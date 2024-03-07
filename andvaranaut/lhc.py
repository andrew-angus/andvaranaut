#!/bin/python3

import numpy as np
import scipy.stats as st
from time import time as stopwatch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy
from functools import partial
from andvaranaut.core import _core
from matplotlib import ticker
from netCDF4 import *
from scipy.stats import qmc

# Latin hypercube sampler and propagator, inherits core
class LHC(_core):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.x = np.empty((0,self.nx))
    self.y = np.empty((0,self.ny))

  # Add n samples to current via latin hypercube sampling
  def sample(self,nsamps,seed=None):
    if not isinstance(nsamps,int) or (nsamps < 1):
      raise Exception("Error: nsamps argument must be an integer > 0") 
    if self.verbose:
      print(f'Evaluating {nsamps} latin hypercube samples...')
    xsamps = self.__latin_sample(nsamps,seed)
    if self.constraints is not None:
      xsamps = self._core__check_constraints(xsamps)
    xsamps,ysamps = self._core__vector_solver(xsamps)

    # Add new evaluations to original data arrays
    self.x = np.r_[self.x,xsamps]
    self.y = np.r_[self.y,ysamps]
    self.nsamp = len(self.x)

  # Produce latin hypercube samples from input distributions
  def __latin_sample(self,nsamps,seed=None):
    #points = latin_random(nsamps,self.nx,seed)
    sampler = qmc.LatinHypercube(d=self.nx,optimization="random-cd")
    points = sampler.random(n=nsamps)
    xsamps = np.zeros_like(points)
    for j in range(self.nx):
      xsamps[:,j] = self.priors[j].ppf(points[:,j])
    return xsamps

  # Delete n samples by selected method
  def del_samples(self,ndels=None,method='coarse_lhc',idx=None):
    self.__del_samples(ndels,method,idx,returns=False)
    self.nsamp = len(self.x)

  # Private method with more flexibility for extension in child classes
  def __del_samples(self,ndels,method,idx,returns):
    # Delete samples by proximity to coarse LHC of size (ndels,nx)
    if method == 'coarse_lhc':
      if not isinstance(ndels,int) or ndels < 1:
        raise Exception("Error: must specify positive int for ndels")
      xsamps = self.__latin_sample(ndels)
      dmins = np.zeros(ndels,dtype=np.intc)
      for i in range(ndels):
        dis = np.zeros(self.nsamp-i)
        for j in range(self.nsamp-i):
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
      left = self.nsamp-ndels
      a = np.arange(0,self.nsamp)
      inds = np.random.choice(a,size=left,replace=False)
      self.x = self.x[inds,:]
      self.y = self.y[inds,:]
      if returns:
        return inds
    # Delete samples at specified indexes
    elif method == 'specific':
      if not isinstance(idx,(int,list)):
        raise Exception("Error: must specify int or list of ints for idx")
      mask = np.ones(self.nsamp, dtype=bool)
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
    # Checks that args are 2D numpy float arrays of correct nx/ny
    if not isinstance(x,np.ndarray) or len(x.shape) != 2 \
        or x.dtype != 'float64' or x.shape[1] != self.nx:
      raise Exception(\
          "Error: Setting data requires a 2d numpy array of float64 inputs")
    if not isinstance(y,np.ndarray) or len(y.shape) != 2 \
        or y.dtype != 'float64' or y.shape[1] != self.ny:
      raise Exception(\
          "Error: Setting data requires a 2d numpy array of float64 outputs")
    # Also check if x data within input distribution interval
    for i in range(self.nx):
      intv = self.priors[i].interval(1.0)
      if not all(x[:,i] >= intv[0]) or not all(x[:,i] <= intv[1]):
        raise Exception(\
            "Error: provided x data must fit within provided input distribution ranges.")
    self.x = x
    self.y = y
    self.nsamp = len(x)

  # Saves key data to netcdf file
  def save_netcdf(self,fname):
    
    f = Dataset(fname,'w')

    # Create dimensions
    inps = f.createDimension('inputs',self.nx)
    outs = f.createDimension('outputs',self.ny)
    samps = f.createDimension('samples',self.x.shape[0])

    # Create variables
    xsamps = f.createVariable('input_samples','f8',('samples','inputs'))
    ysamps = f.createVariable('output_samples','f8',('samples','outputs'))

    # Write variables
    xsamps[:,:] = self.x
    ysamps[:,:] = self.y
    f.close()

  # Loads key data to netcdf file
  def load_netcdf(self,fname):
    
    f = Dataset(fname,'r')

    self.x = f.variables['input_samples'][:,:]
    self.y = f.variables['output_samples'][:,:]

    f.close()

# Inherit from LHC class and add data conversion methods
class _surrogate(LHC):
  def __init__(self,xconrevs=None,yconrevs=None,**kwargs):
    # Call LHC init, then validate and set now data conversion/reversion attributes
    super().__init__(**kwargs)
    self.xc = copy.deepcopy(self.x)
    self.yc = copy.deepcopy(self.y)
    self.__conrev_check(xconrevs,yconrevs)
  
  # Update sampling method to include data conversion
  def sample(self,nsamps,seed=None):
    super().sample(nsamps,seed)
    self.__con(len(self.x))

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
    returned = super()._LHC__del_samples(ndels,method,idx,returns=True)
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

  # Allow for changing conversion/reversion methods
  def change_xconrevs(self,xconrevs=None):
    # Check and set new lists, then update converted datasets
    self.__conrev_check(xconrevs,yconrevs=self.yconrevs)
    for i in range(self.nx):
      self.xc[:,i] = self.xconrevs[i].con(self.x[:,i])

  # Allow for changing conversion/reversion methods
  def change_yconrevs(self,yconrevs=None):
    # Check and set new lists, then update converted datasets
    self.__conrev_check(self.xconrevs,yconrevs)
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
    self.__con(self.nsamp)

  # Inherit and extend y_dist to have dist by surrogate predictions
  def y_dist(self,mode='hist_kde',nsamps=None,return_data=False,surrogate=True,predictfun=None):
    # Allow for use of surrogate evaluations or underlying datasets
    if surrogate:
      xsamps = super()._LHC__latin_sample(nsamps)
      xcons = np.zeros((nsamps,self.nx))
      for i in range(self.nx):
        xcons[:,i] = self.xconrevs[i].con(xsamps[:,i])
      ypreds = predictfun(xcons)
      yrevs = np.zeros((nsamps,self.ny))
      for i in range(self.ny):
        yrevs[:,i] = self.yconrevs[i].rev(ypreds[:,i])
      amax = np.argmax(ypreds)
      idx = (amax//self.ny,amax%self.ny)
      super()._LHC__y_dist(yrevs,mode)
      if return_data:
        return xsamps,yrevs
    elif not surrogate:
      super().y_dist(mode)
    else:
      raise Exception("Error: surrogate argument must be of type bool")

# Class to replace None types in surrogate conrev arguments
class _none_conrev:
  def con(self,x):
   return x 
  def rev(self,x):
   return x 
