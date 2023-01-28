#!/bin/python3

import warnings
from scipy.special import logit as lg
import numpy as np
from functools import partial
import scipy.stats as st
import os
import copy
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer
import pytensor.tensor as pt

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
def log1p_con(y):
  return np.log1p(y)
# Convert positive values to unbounded with logarithm
def log10_con(y):
  return np.log10(y)
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
# Normalise by provided mean and std deviation
def meanstd_con(y,mean,std):
  return (y-mean)/std
# Convert by quantiles
def quantile_con(y,qt):
  return qt.transform(y.reshape(-1,1))[:,0]
# Revert by interquartile range
def robust_con(y,rs):
  return rs.transform(y.reshape(-1,1))[:,0]
# Revert by yeo-johnson transform
def powerT_con(y,pt):
  return pt.transform(y.reshape(-1,1))[:,0]

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
def log1p_rev(y):
  return np.expm1(y)
# Revert logarithm with power
def log10_rev(y):
  return np.power(10,y)
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
# Revert by mean and std deviation
def meanstd_rev(y,mean,std):
  return y*std + mean
# Revert by quantiles
def quantile_rev(y,qt):
  return qt.inverse_transform(y.reshape(-1,1))[:,0]
# Revert by interquartile range
def robust_rev(y,rs):
  return rs.inverse_transform(y.reshape(-1,1))[:,0]
# Revert by yeo-johnson transform
def powerT_rev(y,pt):
  return pt.inverse_transform(y.reshape(-1,1))[:,0]

# Define class wrappers for matching sets of conversions and reversions
# Also allows a standard format for use in surrogates without worrying about function arguments
class normal:
  def __init__(self,dist):
    self.con = partial(std_normal,dist=dist)
    self.rev = partial(normal_rev,dist=dist)
class logit_logistic:
  def __init__(self,dist):
    self.con = partial(logit,dist=dist)
    self.rev = partial(logistic,dist=dist)
class probit:
  def __init__(self,dist):
    self.con = partial(probit_con,dist=dist)
    self.rev = partial(probit_rev,dist=dist)
class cdf:
  def __init__(self,dist):
    self.con = partial(cdf_con,dist=dist)
    self.rev = partial(cdf_rev,dist=dist)
class nonneg:
  def __init__(self):
    self.con = nonneg_con
    self.rev = nonneg_rev
class log1p:
  def __init__(self):
    self.con = log1p_con
    self.rev = log1p_rev
class log10:
  def __init__(self):
    self.con = log10_con
    self.rev = log10_rev
class normalise:
  def __init__(self,fac):
    self.con = partial(normalise_con,fac=fac)
    self.rev = partial(normalise_rev,fac=fac)
class quantile:
  def __init__(self,x,mode='normal'):
    self.mode = mode
    self.qt = QuantileTransformer(output_distribution=mode)
    self.qt.fit(x.reshape(-1,1))
    self.con = partial(quantile_con,qt=self.qt)
    self.rev = partial(quantile_rev,qt=self.qt)
class robust:
  def __init__(self,x):
    self.rs = RobustScaler()
    self.rs.fit(x.reshape(-1,1))
    self.con = partial(robust_con,rs=self.rs)
    self.rev = partial(robust_rev,rs=self.rs)
class powerT:
  def __init__(self,x,method='yeo-johnson'):
    self.method = method
    self.pt = PowerTransformer(method=method)
    self.pt.fit(x.reshape(-1,1))
    lamb = self.pt.lambdas_[0]
    self.pt.lambdas_[0] = np.minimum(np.maximum(-0.01,lamb),1.0)
    self.con = partial(powerT_con,pt=self.pt)
    self.rev = partial(powerT_rev,pt=self.pt)
class logarithm:
  def __init__(self):
    pass
  def con(self,y):
    return np.log(y)
  def rev(self,y):
    return np.exp(y)
  def der(self,y):
    return 1/y
  def conmc(self,y,rvs):
    return pt.log(y)
class affine:
  def __init__(self,a,b):
    self.a = a
    self.b = b
    if not self.b > 0.0:
      raise Exception('Parameter b must be positive')
    self.default_priors = [st.norm(),st.norm()]
  def con(self,y):
    return self.a + self.b*y
  def rev(self,y):
    return (y-self.a)/self.b
  def der(self,y):
    return self.b*np.power(y,0)
  def conmc(self,y,rvs):
    return rvs[0] + rvs[1]*y
class meanstd(affine):
  def __init__(self,y):
    mean = np.mean(y)
    std = np.std(y)
    self.a = -mean/std
    self.b = 1/std
  def conmc(self,y,rvs):
    return self.con(y)
class maxmin(affine):
  def __init__(self,x,centred=False,safety=0.0):
    xmin = np.min(x)*(1-safety)
    xmax = np.max(x)*(1+safety)
    xminus = xmax-xmin
    xplus = xmax+xmin
    if centred:
      self.a = -xplus/xminus
      self.b = 2/xminus
    else:
      self.a = -xmin/xminus
      self.b = 1/xminus
  def conmc(self,y,rvs):
    return self.con(y)
class uniform(affine):
  def __init__(self,dist):
    self.con = partial(std_uniform,dist=dist)
    self.rev = partial(uniform_rev,dist=dist)
    intv = dist.interval(1.0)
    span = intv[1]-intv[0]
    self.a = -intv[0]/span
    self.b = 1/span
  def conmc(self,y,rvs):
    return self.con(y)
class arcsinh:
  def __init__(self,a,b,c,d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.default_priors = [st.norm(),st.norm(),st.norm(),st.norm()]
    if not self.b > 0.0:
      raise Exception('Parameter b must be positive')
    if not self.d > 0.0:
      raise Exception('Parameter d must be positive')
  def con(self,y):
    return self.a + self.b*np.arcsinh((y-self.c)/self.d)
  def rev(self,y):
    return self.c + self.d*np.sinh((y-self.a)/self.b)
  def der(self,y):
    return self.b/np.sqrt(np.power(self.d,2)+np.power(y-self.c,2))
  def conmc(self,y,rvs):
    return rvs[0] + rvs[1]*pt.arcsinh((y-rvs[2])/rvs[3])
# Box cox with lambad defined such that prior peak at 0 gives (almost) identity transform
class boxcox:
  def __init__(self,lamb):
    self.lamb = lamb
    self.default_priors = [st.norm(loc=0)]
  def con(self,y):
    lambp = self.lamb + 1
    return (np.sign(y)*np.power(np.abs(y),lambp)-1)/lambp
  def rev(self,y):
    lambp = self.lamb + 1
    term = y*lambp+1
    return np.sign(term)*np.power(np.abs(term),1/lambp)
  def der(self,y):
    return np.power(np.abs(y),self.lamb)
  def conmc(self,y,rvs):
    lambp = rvs[0] + 1
    return (pt.sgn(y)*pt.power(pt.abs(y),lambp)-1)/lambp
# Box cox as above but auto fitted with scikit-learn
class boxcoxf:
  def __init__(self,y):
    pt = PowerTransformer(method='box-cox',standardize=False)
    pt.fit(y.reshape(-1,1))
  def con(self,y):
    lambp = self.lamb + 1
    return (np.sign(y)*np.power(np.abs(y),lambp)-1)/lambp
  def rev(self,y):
    lambp = self.lamb + 1
    term = y*lambp+1
    return np.sign(term)*np.power(np.abs(term),1/lambp)
  def der(self,y):
    return np.power(np.abs(y),self.lamb)
  def conmc(self,y,rvs):
    lambp = self.lamb + 1
    return (pt.sgn(y)*pt.power(pt.abs(y),lambp)-1)/lambp
class sinharcsinh:
  def __init__(self,a,b):
    self.a = a
    self.b = b
    if not self.b > 0.0:
      raise Exception('Parameter b must be positive')
    self.default_priors = [st.norm(),st.norm()]
  def con(self,y):
    return np.sinh(self.b*np.arcsinh(y)-self.a)
  def rev(self,y):
    return np.sinh((np.arcsinh(y)+self.a)/self.b)
  def der(self,y):
    return self.b*np.cosh(self.b*np.arcsinh(y)-self.a)/np.sqrt(1+np.power(y,2))
  def conmc(self,y,rvs):
    return pt.sinh(rvs[1]*pt.arcsinh(y)-rvs[0])
class sal:
  def __init__(self,a,b,c,d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    if not self.b > 0.0:
      raise Exception('Parameter b must be positive')
    if not self.d > 0.0:
      raise Exception('Parameter d must be positive')
    self.default_priors = [st.norm(),st.norm(),st.norm(),st.norm()]
  def con(self,y):
    return self.a + self.b*np.sinh(self.d*np.arcsinh(y)-self.c)
  def rev(self,y):
    return np.sinh((np.arcsinh((y-self.a)/self.b)+self.c)/self.d)
  def der(self,y):
    return self.b*self.d*np.cosh(self.d*np.arcsinh(y)-self.c)/np.sqrt(1+np.power(y,2))
  def conmc(self,y,rvs):
    return rvs[0] + rvs[1]*pt.sinh(rvs[3]*pt.arcsinh(y)-rvs[2])

# Input warping with kumaraswamy distribution
class kumaraswamy:
  def __init__(self,a,b):
    self.a = a
    self.b = b
    if not self.a > 0.0:
      raise Exception('Parameter a must be positive')
    if not self.b > 0.0:
      raise Exception('Parameter b must be positive')
    self.default_priors = [st.norm(),st.norm()]
  def con(self,x):
    return 1 - np.power(1-np.power(x,self.a),self.b)
  def rev(self,x):
    return np.power(1-np.power(1-x,1/self.b),1/self.a)
  def der(self,x):
    return self.a*self.b*np.power(x,self.a-1)*np.power(1-np.power(x,self.a),self.b-1)
  def conmc(self,x,rvs):
    return 1 - pt.power(1-pt.power(x,rvs[0]),rvs[1])

# Composite warping class
class wgp:
  def __init__(self,warpings,params,y=None,xdist=None):
    allowed = ['affine','logarithm','arcsinh','boxcox','sinharcsinh','sal', \
               'meanstd','boxcoxf','uniform','maxmin','kumaraswamy']
    self.warping_names = warpings
    self.warpings = []
    self.params = params
    self.pid = np.zeros(len(warpings),dtype=np.int32)
    self.pos = np.zeros(len(params),dtype=np.bool_)
    self.default_priors = []
    pc = 0
    pidc = 0
    if y is not None:
      yc = copy.deepcopy(y)
    # Fill self.warpings with conrev classes and \
    # self.pid with the starting index in params for each class
    for i in warpings:
      if i not in allowed:
        raise Exception(f'Only {allowed} classes allowed')
      if i == 'affine':
        self.warpings.append(affine(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
      elif i == 'logarithm':
        self.warpings.append(logarithm())
      elif i == 'arcsinh':
        self.warpings.append(arcsinh(params[pc],params[pc+1],params[pc+2],params[pc+3]))
        self.pos[pc:pc+4] = np.array([False,True,False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 4
      elif i == 'boxcox':
        self.warpings.append(boxcox(lamb=params[pc]))
        self.pos[pc:pc+1] = np.array([False],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 1
      if i == 'sinharcsinh':
        self.warpings.append(sinharcsinh(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
      elif i == 'sal':
        self.warpings.append(sal(params[pc],params[pc+1],params[pc+2],params[pc+3]))
        self.pos[pc:pc+4] = np.array([False,True,False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 4
      if i == 'kumaraswamy':
        self.warpings.append(kumaraswamy(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([True,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
      elif i == 'meanstd':
        if y is None:
          raise Exception('Must supply y array to use meanstd')
        self.warpings.append(meanstd(yc))
      elif i == 'boxcoxf':
        if y is None:
          raise Exception('Must supply y array to use fitted box cox')
        self.warpings.append(boxcoxf(y=yc))
      elif i == 'uniform':
        if xdist is None:
          raise Exception('Must supply x distribution to use uniform')
        self.warpings.append(uniform(xdist))
      elif i == 'maxmin':
        if y is None:
          raise Exception('Must supply y array to use maxmin')
        self.warpings.append(maxmin(yc))
      self.pid[pidc] = pc
      pidc += 1
      if y is not None:
        yc = self.warpings[-1].con(yc)
    self.np = pc
     
  def con(self,y):
    res = y
    for i in self.warpings:
      res = i.con(res)
    return res

  def rev(self,y):
    res = y
    for i in reversed(self.warpings):
      res = i.rev(res)
    return res

  def der(self,y):
    res = np.ones_like(y)
    x = copy.deepcopy(y)
    for i in self.warpings:
      res *= i.der(x)
      x[:,0] = i.con(x[:,0])
    return res

  def conmc(self,y,rvs):
    res = y
    rc = 0
    for i,j in enumerate(self.warpings):
      res = j.conmc(res,rvs[rc:self.pid[i]])
      rc += (self.pid[i]-rc)
    return res