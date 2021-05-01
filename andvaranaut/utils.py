#!/bin/python3

import pickle
from scipy.special import logit as lg
import numpy as np
from functools import partial

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
def logit(x,dists):
  # Convert uniform distribution samples to standard uniform [0,1]
  # and logit transform to unbounded range 
  x01 = std_uniform(x,dists)
  x = __logit(x01)
  return x
# Convert uniform dist samples into standard uniform 
def std_uniform(x,dists):
  x01 = np.zeros_like(x)
  for i in range(len(dists)):
    intv = dists[i].interval(1.0)
    x01[:,i] = (x[:,i]-intv[0])/(intv[1]-intv[0])
  return x01
# Convert normal dist samples into standard normal
def std_normal(x,dists):
  xs = np.zeros_like(x)
  for i in range(len(dists)):
    xs[:,i] = (x[:,i]-dists[i].mean())/dists[i].std()
  return xs
# Convert positive values to unbounded with logarithm
def log_con(y):
  return np.log10(y)
# Convert non-negative to unbounded, via intermediate [0,1]
def nonneg_con(y):
  y01 = y/(1+y)
  return  __logit(y)

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
def logistic(x,dists):
  x01 = __logistic(x)
  x = uniform_rev(x01,dists)
  return x
# Revert to original uniform distributions
def uniform_rev(x01,dists):
  x = np.zeros_like(x01)
  for i in range(len(dists)):
    intv = dists[i].interval(1.0)
    x[:,i] = x01[:,i]*(intv[1]-intv[0])+intv[0]
  return x
# Revert to original uniform distributions
def normal_rev(xs,dists):
  x = np.zeros_like(xs)
  for i in range(len(dists)):
    x[:,i] = xs[:,i]*dists[i].std()+dists[i].mean()
  return x
# Revert logarithm with power
def log_rev(y):
  return np.power(10,y)
# Revert unbounded to non-negative, via intermediate [0,1]
def nonneg_rev(y):
  y01 = __logistic(y)
  return  y01/(1-y01)

# Define class wrappers for matching sets of conversions and reversions
# Also allows a standard format for use in surrogates without worrying about function arguments
class normal:
  def __init__(self,dists):
    self.con = partial(std_normal,dists=dists)
    self.rev = partial(normal_rev,dists=dists)
class uniform:
  def __init__(self,dists):
    self.con = partial(std_uniform,dists=dists)
    self.rev = partial(uniform_rev,dists=dists)
class logit_logistic:
  def __init__(self,dists):
    self.con = partial(logit,dists=dists)
    self.rev = partial(logistic,dists=dists)
class nonneg:
  def __init__(self):
    self.con = nonneg_con
    self.rev = nonneg_rev
class logarithm:
  def __init__(self):
    self.con = log_con
    self.rev = log_rev


    
