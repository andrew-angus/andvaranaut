#!/bin/python3

import pickle
from scipy.special import logit as lg
import numpy as np
from functools import partial
import scipy.stats as st

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
    self.prior = st.norm()
class uniform:
  def __init__(self,dist):
    self.con = partial(std_uniform,dist=dist)
    self.rev = partial(uniform_rev,dist=dist)
    self.prior = st.uniform()
class logit_logistic:
  def __init__(self,dist):
    self.con = partial(logit,dist=dist)
    self.rev = partial(logistic,dist=dist)
    self.prior = st.logistic()
class probit:
  def __init__(self,dist):
    self.con = partial(probit_con,dist=dist)
    self.rev = partial(probit_rev,dist=dist)
    self.prior = st.norm()
class cdf:
  def __init__(self,dist):
    self.con = partial(cdf_con,dist=dist)
    self.rev = partial(cdf_rev,dist=dist)
    self.prior = st.uniform()
class nonneg:
  def __init__(self):
    self.con = nonneg_con
    self.rev = nonneg_rev
    self.prior = None
class logarithm:
  def __init__(self):
    self.con = log_con
    self.rev = log_rev
    self.prior = None
class normalise:
  def __init__(self,fac):
    self.con = partial(normalise_con,fac=fac)
    self.rev = partial(normalise_rev,fac=fac)
    self.prior = None
