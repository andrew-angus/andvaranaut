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
from pytensor import shared as as_pt

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
  def conmc(self,y):
    return pt.log(y)
  def revmc(self,y):
    return pt.exp(y)
  def dermc(self,y):
    return 1/y
class affine:
  def __init__(self,a,b):
    self.a = a
    self.b = b
    try:
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
    except:
      pass
    self.default_priors = [st.norm(),st.norm()]
  def con(self,y):
    return self.a + self.b*y
  def rev(self,y):
    return (y-self.a)/self.b
  def der(self,y):
    return self.b*np.ones_like(y)
  def conmc(self,y):
    return self.con(y)
  def revmc(self,y):
    return self.rev(y)
  def dermc(self,y):
    return self.b*pt.ones_like(y)
class meanstd(affine):
  def __init__(self,y,mode='numpy'):
    if mode == 'numpy':
      mean = np.mean(y)
      std = np.std(y)
    else:
      mean = pt.mean(y)
      std = pt.std(y)
    self.a = -mean/std
    self.b = 1/std
class minshift(affine):
  def __init__(self,y,mode='numpy',safety=1000):
    if mode == 'numpy':
      mini = np.min(y)
    else:
      mini = pt.min(y)
    self.a = -mini*safety
    self.b = 1.0
class stddev(affine):
  def __init__(self,y,mode='numpy'):
    if mode == 'numpy':
      std = np.std(y)
    else:
      std = pt.std(y)
    self.a = 0
    self.b = 1/std
class stdshift(affine):
  def __init__(self,a,y,mode='numpy'):
    if mode == 'numpy':
      std = np.std(y)
    else:
      std = pt.std(y)
    self.a = a
    self.b = 1/std
    self.default_priors = [st.norm()]
class maxmin(affine):
  def __init__(self,x,centred=False,safety=0.01,mode='numpy'):
    if mode == 'numpy':
      xmin = np.min(x)
      xmax = np.max(x)
    else:
      xmin = pt.min(x)
      xmax = pt.max(x)
    xminus = (xmax-xmin)/(1-2*safety)
    xplus = xmax+xmin
    if centred:
      self.a = -xplus/xminus
      self.b = 2/xminus
    else:
      self.a = -xmin/xminus+safety
      self.b = 1/xminus
class uniform(affine):
  def __init__(self,dist,safety=1e-10):
    #self.con = partial(std_uniform,dist=dist)
    #self.rev = partial(uniform_rev,dist=dist)
    intv = dist.interval(1.0)
    xminus = (intv[1]-intv[0])/(1-2*safety)
    self.a = -intv[0]/xminus + safety
    self.b = 1/xminus
class arcsinh:
  def __init__(self,a,b,c,d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.default_priors = [st.norm(),st.norm(),st.norm(),st.norm()]
    try:
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
      if not self.d > 0.0:
        raise Exception('Parameter d must be positive')
    except:
      pass
  def con(self,y):
    return self.a + self.b*np.arcsinh((y-self.c)/self.d)
  def rev(self,y):
    return self.c + self.d*np.sinh((y-self.a)/self.b)
  def der(self,y):
    return self.b/np.sqrt(np.power(self.d,2)+np.power(y-self.c,2))
  def conmc(self,y):
    return self.a + self.b*pt.arcsinh((y-self.c)/self.d)
  def revmc(self,y):
    return self.c + self.d*pt.sinh((y-self.a)/self.b)
  def dermc(self,y):
    return self.b/pt.sqrt(pt.power(self.d,2)+pt.power(y-self.c,2))
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
  def conmc(self,y):
    lambp = self.lamb + 1
    return (pt.sgn(y)*pt.power(pt.abs(y),lambp)-1)/lambp
  def revmc(self,y):
    lambp = self.lamb + 1
    term = y*lambp+1
    return pt.sgn(term)*pt.power(pt.abs(term),1/lambp)
  def dermc(self,y):
    return pt.power(pt.abs(y),self.lamb)
# Box cox as above but auto fitted with scikit-learn
class boxcoxf(boxcox):
  def __init__(self,y):
    powt = PowerTransformer(method='box-cox',standardize=False)
    powt.fit(y.reshape(-1,1))
    self.lamb = powt.lambdas_[0]
class sinharcsinh:
  def __init__(self,a,b):
    self.a = a
    self.b = b
    try:
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
    except:
      pass
    self.default_priors = [st.norm(),st.norm()]
  def con(self,y):
    return np.sinh(self.b*np.arcsinh(y)-self.a)
  def rev(self,y):
    return np.sinh((np.arcsinh(y)+self.a)/self.b)
  def der(self,y):
    return self.b*np.cosh(self.b*np.arcsinh(y)-self.a)/np.sqrt(1+np.power(y,2))
  def conmc(self,y):
    return pt.sinh(self.b*pt.arcsinh(y)-self.a)
  def revmc(self,y):
    return pt.sinh((pt.arcsinh(y)+self.a)/self.b)
  def dermc(self,y):
    return self.b*pt.cosh(self.b*pt.arcsinh(y)-self.a)/pt.sqrt(1+pt.power(y,2))
class sal:
  def __init__(self,a,b,c,d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    try:
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
      if not self.d > 0.0:
        raise Exception('Parameter d must be positive')
    except:
      pass
    self.default_priors = [st.norm(),st.norm(),st.norm(),st.norm()]
  def con(self,y):
    return self.c + self.d*np.sinh(self.b*np.arcsinh(y)-self.a)
  def rev(self,y):
    return np.sinh((np.arcsinh((y-self.c)/self.d)+self.a)/self.b)
  def der(self,y):
    return self.b*self.d*np.cosh(self.b*np.arcsinh(y)-self.a)/np.sqrt(1+np.power(y,2))
  def conmc(self,y):
    return self.c + self.d*pt.sinh(self.b*pt.arcsinh(y)-self.a)
  def revmc(self,y):
    return pt.sinh((pt.arcsinh((y-self.c)/self.d)+self.a)/self.b)
  def dermc(self,y):
    return self.b*self.d*pt.cosh(self.b*pt.arcsinh(y)-self.a)/pt.sqrt(1+pt.power(y,2))

# Input warping with kumaraswamy distribution
class kumaraswamy:
  def __init__(self,a,b):
    self.a = a
    self.b = b
    try:
      if not self.a > 0.0:
        raise Exception('Parameter a must be positive')
      if not self.b > 0.0:
        raise Exception('Parameter b must be positive')
    except:
      pass
    self.default_priors = [st.norm(),st.norm()]
  def con(self,x):
    return 1 - np.power(1-np.power(x,self.a),self.b)
  def rev(self,x):
    return np.power(1-np.power(1-x,1/self.b),1/self.a)
  def der(self,x):
    return self.a*self.b*np.power(x,self.a-1)*np.power(1-np.power(x,self.a),self.b-1)
  def conmc(self,x):
    return 1 - pt.power(1-pt.power(x,self.a),self.b)
  def revmc(self,x):
    return pt.power(1-pt.power(1-x,1/self.b),1/self.a)
  def dermc(self,x):
    return self.a*self.b*pt.power(x,self.a-1)*pt.power(1-pt.power(x,self.a),self.b-1)

# Transform that preserves a mapping of zero to zero
# Important with delta learning
class preserve_zero(affine):
  def __init__(self,y,yzero,mode='numpy'):
    if mode == 'numpy':
      ystd = np.std(y)
    else:
      ystd = pt.std(y)
    self.a = -yzero/ystd
    self.b = 1/ystd

# Composite warping class
class wgp:
  def __init__(self,warpings,params,y=None,xdist=None,mode='numpy'):
    allowed = ['affine','logarithm','arcsinh','boxcox','sinharcsinh','sal', \
               'meanstd','boxcoxf','uniform','maxmin','kumaraswamy','pzero',\
               'stddev','stdshift','minshift']
    self.warping_names = warpings
    self.warpings = []
    self.params = params
    self.pid = np.zeros(len(warpings),dtype=np.int32)
    try:
      self.pos = np.zeros(len(params),dtype=np.bool_)
    except:
      self.pos = np.zeros(len(params.eval()),dtype=np.bool_)
    self.default_priors = []
    pc = 0
    pidc = 0
    yzero = 0.0
    if y is not None:
      if mode == 'numpy':
        yc = copy.deepcopy(y)
      else:
        yc = pt.as_tensor_variable(y)
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
      elif i == 'sinharcsinh':
        self.warpings.append(sinharcsinh(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
      elif i == 'sal':
        self.warpings.append(sal(params[pc],params[pc+1],params[pc+2],params[pc+3]))
        self.pos[pc:pc+4] = np.array([False,True,False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 4
      elif i == 'kumaraswamy':
        self.warpings.append(kumaraswamy(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([True,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
      elif i == 'stdshift':
        if y is None:
          raise Exception('Must supply y array to use stddev')
        self.warpings.append(stdshift(params[pc],yc,mode=mode))
        self.pos[pc] = np.array([False],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 1
      elif i == 'meanstd':
        if y is None:
          raise Exception('Must supply y array to use meanstd')
        self.warpings.append(meanstd(yc,mode=mode))
      elif i == 'minshift':
        if y is None:
          raise Exception('Must supply y array to use minshift')
        self.warpings.append(minshift(yc,mode=mode))
      elif i == 'stddev':
        if y is None:
          raise Exception('Must supply y array to use stddev')
        self.warpings.append(stddev(yc,mode=mode))
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
        self.warpings.append(maxmin(yc,mode=mode))
      elif i == 'pzero':
        if y is None:
          raise Exception('Must supply y array to use pzero')
        self.warpings.append(preserve_zero(yc,yzero,mode=mode))
      self.pid[pidc] = pc
      pidc += 1
      if y is not None:
        if mode == 'numpy':
          yc = self.warpings[-1].con(yc)
          yzero = self.warpings[-1].con(yzero)
        else:
          yc = self.warpings[-1].conmc(yc)
          yzero = self.warpings[-1].conmc(yzero)
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
      x = i.con(x)
    return res

  def conmc(self,y):
    res = y
    for i in self.warpings:
      res = i.conmc(res)
    return res

  def revmc(self,y):
    res = y
    for i in reversed(self.warpings):
      res = i.revmc(res)
    return res

  def dermc(self,y):
    res = pt.ones_like(y)
    x = y
    for i in self.warpings:
      res *= i.dermc(x)
      x = i.conmc(x)
    return res

  """
  def conmc(self,y,rvs):
    res = y
    rc = 0
    for i,j in enumerate(self.warpings):
      res = j.conmc(res,rvs[rc:self.pid[i]])
      rc += (self.pid[i]-rc)
    return res

  def dermc(self,y,rvs):
    res = pt.ones_like(y)
    #x = copy.deepcopy(y)
    #x = y
    rc = 0
    for i,j in enumerate(self.warpings):
      res *= j.dermc(y,rvs[rc:self.pid[i]])
      y = j.conmc(y,rvs[rc:self.pid[i]])
      rc += (self.pid[i]-rc)
    return res
  """

