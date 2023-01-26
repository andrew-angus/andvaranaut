#!/bin/python3

import warnings
import pickle
from scipy.special import logit as lg
import numpy as np
from functools import partial
import scipy.stats as st
import ray
import multiprocessing as mp
from time import time as stopwatch
import os
import copy
from scipy.optimize import differential_evolution,NonlinearConstraint,minimize, Bounds
from design import ihs
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer
import GPy
from GPyOpt.models import GPModel
from GPyOpt.methods import BayesianOptimization

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
#class uniform:
#  def __init__(self,dist):
#    self.con = partial(std_uniform,dist=dist)
#    self.rev = partial(uniform_rev,dist=dist)
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
class logarithm:
  def __init__(self):
    pass
  def con(self,y):
    return np.log(y)
  def rev(self,y):
    return np.exp(y)
  def der(self,y):
    return 1/y
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
#class meanstd:
  #def __init__(self,x):
    #self.mean = np.mean(x)
    #self.std = np.std(x)
    #self.con = partial(meanstd_con,mean=self.mean,std=self.std)
    #self.rev = partial(meanstd_rev,mean=self.mean,std=self.std)
#class minmax:
  #def __init__(self,x,centred=False):
  #  self.centred = centred
  #  self.min = np.min(x)
  #  self.max = np.max(x)
  #def con(self,x):
  #  if self.centred:
  #    return (2*x - (self.max+self.min))/(self.max-self.min)
  #  else:
  #    return (x - self.min)/(self.max-self.min)
  #def rev(self,x):
  #  if self.centred:
  #    return (x*(self.max-self.min)+(self.max+self.min))/2
  #  else:
  #    return x*(self.max-self.min)+self.min
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
class meanstd(affine):
  def __init__(self,y):
    mean = np.mean(y)
    std = np.std(y)
    self.a = -mean/std
    self.b = 1/std
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
class uniform(affine):
  def __init__(self,dist):
    self.con = partial(std_uniform,dist=dist)
    self.rev = partial(uniform_rev,dist=dist)
    intv = dist.interval(1.0)
    span = intv[1]-intv[0]
    self.a = -intv[0]/span
    self.b = 1/span
class arcsinh:
  def __init__(self,a,b,c,d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.default_priors = [st.norm(),st.norm(),st.norm(),st.norm(loc=1)]
    if not self.b > 0.0:
      raise Exception('Parameter b must be positive')
  def con(self,y):
    return self.a + self.b*np.arcsinh((y-self.c)/self.d)
  def rev(self,y):
    return self.c + self.d*np.sinh((y-self.a)/self.b)
  def der(self,y):
    return self.b/np.sqrt(np.power(self.d,2)+np.power(y-self.c,2))
class boxcox:
  def __init__(self,y=None,lamb=None):
    self.pt = PowerTransformer(method='box-cox',standardize=False)
    if lamb is None:
      if y is None:
        raise Exception(\
            'Error: Must provide either an array for fitting or lambda parameter')
      else:
        self.pt.fit(y.reshape(-1,1))
    else:
      self.pt.lambdas_ = np.array([lamb])
    self.default_priors = [st.norm(loc=1)]
  def con(self,y):
    return self.pt.transform(y.reshape(-1,1))[:,0]
  def rev(self,y):
    return self.pt.inverse_transform(y.reshape(-1,1))[:,0]
  def der(self,y):
    return np.power(np.abs(y),self.pt.lambdas_[0]-1)
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


# Composite warping class
class wgp:
  def __init__(self,warpings,params,y=None,xdist=None):
    allowed = ['affine','logarithm','arcsinh','boxcox','sinharcsinh','sal', \
               'meanstd','boxcoxf','uniform','maxmin','kumaraswamy']
    self.warping_names = warpings
    self.warpings = []
    self.params = params
    self.pid = np.zeros(len(warpings))
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
        self.pid[pidc] = pc
        self.warpings.append(affine(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
        pidc += 1
      elif i == 'logarithm':
        self.pid[pidc] = pc
        self.warpings.append(logarithm())
        pidc += 1
      elif i == 'arcsinh':
        self.pid[pidc] = pc
        self.warpings.append(arcsinh(params[pc],params[pc+1],params[pc+2],params[pc+3]))
        self.pos[pc:pc+4] = np.array([False,True,False,False],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 4
        pidc += 1
      elif i == 'boxcox':
        self.pid[pidc] = pc
        self.warpings.append(boxcox(lamb=params[pc]))
        self.pos[pc:pc+1] = np.array([False],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 1
        pidc += 1
      if i == 'sinharcsinh':
        self.pid[pidc] = pc
        self.warpings.append(sinharcsinh(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
        pidc += 1
      elif i == 'sal':
        self.pid[pidc] = pc
        self.warpings.append(sal(params[pc],params[pc+1],params[pc+2],params[pc+3]))
        self.pos[pc:pc+4] = np.array([False,True,False,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 4
        pidc += 1
      if i == 'kumaraswamy':
        self.pid[pidc] = pc
        self.warpings.append(kumaraswamy(params[pc],params[pc+1]))
        self.pos[pc:pc+2] = np.array([True,True],dtype=np.bool_)
        self.default_priors.extend(self.warpings[-1].default_priors)
        pc += 2
        pidc += 1
      elif i == 'meanstd':
        if y is None:
          raise Exception('Must supply y array to use meanstd')
        self.pid[pidc] = pc
        self.warpings.append(meanstd(yc))
        pidc += 1
      elif i == 'boxcoxf':
        if y is None:
          raise Exception('Must supply y array to use fitted box cox')
        self.pid[pidc] = pc
        self.warpings.append(boxcox(y=yc))
        pidc += 1
      elif i == 'uniform':
        if xdist is None:
          raise Exception('Must supply x distribution to use uniform')
        self.pid[pidc] = pc
        self.warpings.append(uniform(xdist))
        pidc += 1
      elif i == 'maxmin':
        if y is None:
          raise Exception('Must supply y array to use maxmin')
        self.pid[pidc] = pc
        self.warpings.append(maxmin(yc))
        pidc += 1
      if y is not None:
        yc = self.warpings[-1].con(yc)
     
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

# Core class which runs target function
class _core():
  def __init__(self,nx,ny,priors,target,parallel=False,nproc=1,\
      constraints=None,rundir=None,verbose=False):
    # Check inputs
    if (not isinstance(nx,int)) or (nx < 1):
      raise Exception('Error: must specify an integer number of input dimensions > 0') 
    if (not isinstance(ny,int)) or (ny < 1):
      raise Exception('Error: must specify an integer number of output dimensions > 0') 
    if (not isinstance(priors,list)) or (len(priors) != nx):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate priors of length nx') 
    check = 'scipy.stats._distn_infrastructure'
    flags = [not getattr(i,'__module__',None)==check for i in priors]
    if any(flags):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate priors of length nx') 
    if not callable(target):
      raise Exception(\
          'Error: must provide target function which produces output from specified inputs')
    if not isinstance(parallel,bool):
      raise Exception("Error: parallel must be type bool.")
    if not isinstance(nproc,int) or (nproc < 1):
      raise Exception("Error: nproc argument must be an integer > 0")
    assert (nproc <= mp.cpu_count()),\
        "Error: number of processors selected exceeds available."
    if (not isinstance(constraints,dict)) and (constraints is not None):
      raise Exception(\
          f'Error: provided constraints must be a dictionary with keys {keys} and list items.') 
    keys = ['constraints','lower_bounds','upper_bounds']
    if constraints is not None:
      if not all(key in constraints for key in keys):
        raise Exception(\
          f'Error: provided constraints must be a dictionary with keys {keys} and list items.') 
    # Initialise attributes
    self.nx = nx # Input dimensions
    self.ny = ny # Output dimensions
    self.priors = priors # Input distributions (must be scipy)
    self.target = target # Target function which takes X and returns Y
    self.parallel = parallel # Whether to use parallelism wherever possible
    self.nproc = nproc # Number of processors to use if using parallelism
    self.constraints = constraints # List of constraint functions for sampler
    self.verbose = verbose
    self.rundir = 'runs'
    if rundir is not None:
      self.rundir = rundir

  # Method which takes function, and 2D array of inputs
  # Then runs in parallel for each set of inputs
  # Returning 2D array of outputs
  def __parallel_runs(self,inps):

    # Run function in parallel in individual directories    
    if not ray.is_initialized():
      ray.init(num_cpus=self.nproc)
    l = len(inps)
    all_ids = [_parallel_wrap.remote(self.target,self.rundir,inps[i],i) for i in range(l)]

    # Get ids as they complete or fail, give warning on fail
    outs = []; fails = np.empty(0,dtype=np.intc)
    id_order = np.empty(0,dtype=np.intc)
    ids = copy.deepcopy(all_ids)
    lold = l; flag = False
    while lold:
      done_id,ids = ray.wait(ids)
      try:
        outs += ray.get(done_id)
        idx = all_ids.index(done_id[0]) 
        id_order = np.append(id_order,idx)
      except:
        idx = all_ids.index(done_id[0]) 
        id_order = np.append(id_order,idx)
        fails = np.append(fails,idx)
        flag = True
        print(f"Warning: parallel run {idx+1} failed with x values {inps[idx]}.",\
          "\nCheck number of inputs/outputs and whether input ranges are valid.")
      lnew = len(ids)
      if lnew != lold:
        lold = lnew
        if self.verbose:
          print(f'Run is {(l-lold)/l:0.1%} complete.',end='\r')
    if flag:
      ray.shutdown()
    
    # Reshape outputs to 2D array
    oldouts = np.array(outs).reshape((len(outs),self.ny))
    outs = np.zeros_like(oldouts)
    outs[id_order] = oldouts

    return outs, fails

  # Private method which takes array of x samples and evaluates y at each
  def __vector_solver(self,xsamps):
    t0 = stopwatch()
    n_samples = len(xsamps)
    # Create directory for tasks
    if not os.path.isdir(self.rundir):
      os.mkdir(self.rundir)
    # Parallel execution using ray
    if self.parallel:
      ysamps,fails = self.__parallel_runs(xsamps)
      assert ysamps.shape[1] == self.ny, "Specified ny does not match function output"
    # Serial execution
    else:
      ysamps = np.empty((0,self.ny))
      fails = np.empty(0,dtype=np.intc)
      for i in range(n_samples):
        d = os.path.join(self.rundir, f'task{i}')
        if not os.path.isdir(d):
          os.system(f'mkdir {d}')
        os.chdir(d)
        # Keep track of fails but run rest of samples
        try:
          yout = self.target(xsamps[i,:])
        except:
          errstr = f"Warning: Target function evaluation failed at sample {i+1} "+\
              "with x values: " +str(xsamps[i,:])+\
              "\nCheck number of inputs and range of input values valid."
          print(errstr)
          fails = np.append(fails,i)
          os.chdir('../..')
          continue


        # Number of function outputs check and append samples
        try:
          ysamps = np.vstack((ysamps,yout))
        except:
          os.chdir('../..')
          raise Exception("Error: number of target function outputs is not equal to ny")
        os.chdir('../..')
        if self.verbose:
          print(f'Run is {(i+1)/n_samples:0.1%} complete.',end='\r')
    t1 = stopwatch()

    # Remove failed samples
    mask = np.ones(n_samples, dtype=bool)
    mask[fails] = False
    xsamps = xsamps[mask]

    # NaN and inf check
    fails = np.empty(0,dtype=np.intc)
    for i,j in enumerate(ysamps):
      if np.any(np.isnan(j)) or np.any(np.abs(j) == np.inf):
        fails = np.append(fails,i)
        errstr = f"Warning: Target function evaluation returned inf/nan at sample "+\
            "with x values: " +str(xsamps[i,:])+"\nCheck range of input values valid."
        print(errstr)
    mask = np.ones(len(xsamps),dtype=bool)
    mask[fails] = False
    xsamps = xsamps[mask]
    ysamps = ysamps[mask]

    # Final print on time taken
    if self.verbose:
      print()
      print(f'Time taken: {t1-t0:0.2f} s')

    return xsamps, ysamps

  # Core optimizer implementing bounds and constraints
  # Global optisation done either with differential evolution or local minimisation with restarts
  def __opt(self,fun,method,nx,restarts=10,nonself=False,priors=None,**kwargs):
    # Construct constraints object if using
    if not nonself:
      if self.constraints is not None:
        cons = self.constraints['constraints']
        upps = self.constraints['upper_bounds']
        lows = self.constraints['lower_bounds']
        nlcs = tuple(NonlinearConstraint(cons[i],lows[i],upps[i]) for i in range(len(cons)))
      else:
        nlcs = tuple()
      kwargs['constraints'] = nlcs

    # Set bounds with priors if exists
    if priors is not None:
      lbs = np.zeros(nx)
      ubs = np.zeros(nx)
      for j in range(nx):
        lbs[j] = priors[j].ppf(1e-8)
        ubs[j] = priors[j].isf(1e-8)
      kwargs['bounds'] = Bounds(lbs,ubs)

    # Global opt method choice
    verbose = self.verbose
    self.verbose = False
    if method == 'DE':
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = differential_evolution(fun,**kwargs)
    elif method == 'restarts':
      # Add buffer to nlcs to stop overshoot
      if not nonself:
        buff = 1e-6
        for i in nlcs:
          i.lb += buff
          i.ub -= buff
      # Draw starting point samples
      points = (ihs(restarts,nx)\
          -1.0+np.random.rand(restarts,nx))/restarts
      # Scale by bounds or priors
      if priors is not None:
        for j in range(nx):
          points[:,j] = priors[j].ppf(points[:,j])
      else:
        bnds = kwargs['bounds']
        points = self.__bounds_scale(points,nx,bnds)
      # Check against constraints and replace if invalid
      if self.constraints is not None and not nonself:
        points = self.__check_constraints(points)
        npoints = len(points)
        # Add points by random sampling and repeat till all valid
        while npoints != restarts:
          nnew = restarts - npoints
          newpoints = np.random.rand(nnew,nx)
          newpoints = self.__bounds_scale(newpoints,nx,bnds)
          newpoints = self.__check_constraints(newpoints)
          points = np.r_[points,newpoints]
          npoints = len(points)

      # Conduct minimisations
      if self.parallel:
        if not ray.is_initialized():
          ray.init(num_cpus=self.nproc)
        # Switch off further parallelism within minimized function
        self.parallel = False
        try:
          results = ray.get([_minimize_wrap.remote(fun,i,**kwargs) for i in points])
          self.parallel = True
        except:
          self.parallel = True
          ray.shutdown()
          raise Exception
      else:
        results = []
        for i in points:
          with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(fun,i,**kwargs)
          results.append(res)

      # Get best result
      f_vals = np.array([i.fun for i in results])
      f_success = [i.success for i in results]
      if not any(f_success):
        print('Warning: All minimizations unsuccesful')
      elif not all(f_success):
        self.verbose = verbose
        if self.verbose:
          print('Removing failed minimizations...')
        f_success = np.where(np.isnan(f_vals),False,f_success)
        f_vals = f_vals[f_success]
        idx = np.arange(len(results))
        idx = idx[f_success]
        results = [results[i] for i in idx]

      best = np.argmin(f_vals)
      res = results[best]

    elif method == 'BO':
      # Setup domain as list of dictionaries for each variable
      domain = []
      for i in range(nx):
        lb = kwargs['bounds'].lb[i]
        ub = kwargs['bounds'].ub[i]
        xdict = {'name': f'var_{i}','type':'continuous','domain':(lb,ub)}
        domain.append(xdict)

      ## TODO: Constraint setup
      
      # Wrapper around function which inputs 2D arrays
      def fun_wrap(x):
        y = np.zeros((len(x),1))
        for i in range(len(x)):
          y[i,:] = fun(x[i,:])
        return y
      
      # Setup BayesianOptimisation object
      if not nonself:
        kstring = 'GPy.kern.'+self.kernel+\
            '(input_dim=self.nx,variance=1.,lengthscale=1.,ARD=True)'
        kern = eval(kstring)
        bogp = GPModel(kernel=kern,exact_feval=not self.noise,\
            optimize_restarts=restarts,verbose=False,ARD=True)
        bopt = BayesianOptimization(f=fun_wrap,domain=domain,\
            X=self.xc,Y=self.yc,normalize_Y=self.normalise,\
            exact_feval=not self.noise,verbosity=False,\
            initial_design_numdata=0,model=bogp)
      else:
        if 'initial' in kwargs:
          initial = kwargs['initial']
        else:
          initial = 20 
        if 'model_type' in kwargs:
          model_type = kwargs['model_type']
        else:
          model_type = 'GP' 
        bopt = BayesianOptimization(f=fun_wrap,domain=domain,\
            exact_feval=True,verbosity=False,model_type=model_type,\
            initial_design_numdata=initial,ARD=True)

      # Run optimisation
      if 'max_iter' in kwargs:
        bopt.run_optimization(max_iter=kwargs['max_iter'])
      else:
        bopt.run_optimization(max_iter=15)

      xopt = bopt.x_opt
      yopt = bopt.fx_opt

      class res_class:
        def __init__(self,xopt,yopt):
          self.x = xopt
          self.fun = yopt

      res = res_class(xopt,yopt)

    self.verbose = verbose
    return res

  # Check proposed samples against all provided constraints
  def __check_constraints(self,xsamps):
    nsamps0 = len(xsamps)
    mask = np.ones(nsamps0,dtype=bool)
    for i,j in enumerate(xsamps):
      for e,f in enumerate(self.constraints['constraints']):
        flag = True
        res = f(j)
        lower_bounds = self.constraints['lower_bounds'][e]
        upper_bounds = self.constraints['upper_bounds'][e]
        if isinstance(lower_bounds,list):
          for k,l in enumerate(lower_bounds):
            if res[k] < l:
              flag = False
          for k,l in enumerate(upper_bounds):
            if res[k] > l:
              flag = False
        else:
          if res < lower_bounds:
            flag = False
          elif res > upper_bounds:
            flag = False
        mask[i] = flag
        if not flag:
          print(f'Sample {i+1} with x values {j} removed due to invalidaing constraint {e+1}.')
    xsamps = xsamps[mask]
    nsamps1 = len(xsamps)
    if nsamps1 < nsamps0:
      print(f'{nsamps0-nsamps1} samples removed due to violating constraints.')
    return xsamps

  def __bounds_scale(self,points,nx,bnds):
    for i in range(nx):
      points[:,i] *= bnds.ub[i]-bnds.lb[i]
      points[:,i] += bnds.lb[i]
    return points

  # Calculate first derivative with second order central differences
  def __derivative(self,x,fun,idx,eps=1e-6):
    
    # Get shifted input arrays
    xdown = copy.deepcopy(x)
    xup = copy.deepcopy(x)
    xdown[idx] -= eps
    xup[idx] += eps

    # Get function results
    fdown = fun(xdown)
    fup = fun(xup)

    # Calculate derivative
    res = (fup - fdown) / (2*eps)
    return res

  # Calculates gradient vector
  def __grad(self,x,fun,eps=1e-6):
    
    lenx = len(x)
    res = np.zeros(lenx)
    for i in range(lenx):
      res[i] = self.__derivative(x,fun,i,eps)
    return res

  # Calculate hessian matrix
  def __hessian(self,x,fun,eps=1e-6):

    # Compute matrix as jacobian of gradient vector
    lenx = len(x)
    res = np.zeros((lenx,lenx))
    grad = partial(self.__grad,fun=fun,eps=eps)
    for i in range(lenx):
      res[:,i] = self.__derivative(x,grad,i,eps)
    return res 

    # Compute matrix as symmetric derivative of derivative
    lenx = len(x)
    res = np.zeros((lenx,lenx))
    for i in range(lenx):
      for j in range(i):
        div = partial(self.__derivative,fun=fun,eps=eps,idx=i)
        res[i,j] = self.__derivative(x,div,j,eps)
        res[j,i] = res[i,j]
    return res 


# Function which wraps serial function for executing in parallel directories
@ray.remote(max_retries=0)
def _parallel_wrap(fun,rundir,inp,idx):
  d = os.path.join(rundir, f'task{idx}')
  if not os.path.isdir(d):
    os.mkdir(d)
  os.chdir(d)
  res = fun(inp)
  os.chdir('../..')
  return res
 
# Function which wraps serial function for executing in parallel directories
@ray.remote(max_retries=0)
def _minimize_wrap(fun,x0,**kwargs):
  res = minimize(fun,x0,method='SLSQP',**kwargs)
  return res
