#!/bin/python3

from andvaranaut.forward import gp as gp_class
import numpy as np
from scipy.optimize import differential_evolution
import copy

# Maximum a posteriori class
class MAP:
  def __init__(self,nx,ny,dists,target,parallel=False,nproc=1):
    # Check inputs
    if (not isinstance(nx,int)) or (nx < 1):
      raise Exception('Error: must specify an integer number of model input parameters > 0') 
    if (not isinstance(ny,int)) or (ny < 1):
      raise Exception('Error: must specify an integer number of output dimensions > 0') 
    if (not isinstance(priors,list)) or (len(priors) != nx):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate distributions of length nx') 
    check = 'scipy.stats._distn_infrastructure'
    flags = [not getattr(i,'__module__',None)==check for i in dists]
    if any(flags):
      raise Exception(\
          'Error: must provide list of scipy.stats univariate distributions of length nx') 
    if not callable(target):
      raise Exception(\
          'Error: must provide target function which produces output from specified inputs')
    if not isinstance(parallel,bool):
      raise Exception("Error: parallel must be type bool.")
    if not isinstance(nproc,int) or (nproc < 1):
      raise Exception("Error: nproc argument must be an integer > 0")
    assert (nproc <= mp.cpu_count()),\
        "Error: number of processors selected exceeds available."
    # Initialise attributes
    self.nx = nx # Input dimensions
    self.ny = ny # Output dimensions
    self.dists = dists # Input distributions (must be scipy)
    self.x = np.empty((0,nx))
    self.y = np.empty((0,ny))
    self.target = target # Target function which takes X and returns Y
    self.parallel = parallel # Whether to use parallelism wherever possible
    self.nproc = nproc # Number of processors to use if using parallelism

# Quick and dirty maximum a posteriori class using a GP
class gpmap:
  def __init__(self,nx_exp,nx_model,gp):
    # Check inputs
    if (not isinstance(nx_exp,int)) or (nx_exp < 0):
      raise Exception('Error: must specify an integer number of model input parameters => 0') 
    if (not isinstance(nx_model,int)) or (nx_model < 1):
      raise Exception('Error: must specify an integer number of model input parameters > 0') 
    if (not isinstance(gp,gp_class)):
      raise Exception("Error: must provide gp class instance from andvaranaut.forward module")
    if (nx_exp+nx_model != gp.nx):
      raise Exception("Error: nx_exp and nx_model must sum to gp.nx")

    # Initialise attributes
    self.nx_exp = nx_exp # Number of experimental inputs
    self.nx_model = nx_model # Number of model inputs
    self.gp = gp
    self.xopt = None
    self.xcopt = None
    self.yexp = None
    self.ycexp = None

  def set_y(self,y):
    self.xext = np.r_[self.gp.x,np.zeros((len(y),self.gp.nx))]
    self.yext = np.r_[self.gp.y,y]
    self.yexp = y
    for i in range(self.gp.nx):
      self.xext[:,i] = self.gp.xconrevs[i].con(self.xext[:,i])
    for i in range(self.gp.ny):
      self.yext[:,i] = self.gp.yconrevs[i].con(self.yext[:,i])
    self.ycexp = self.yext[-len(y):]

  def log_prior(self,x):
    logps = np.zeros(self.gp.nx)
    for i in range(self.gp.nx):
      logps[i] = self.gp.xconrevs[i].prior.logpdf(x[i])
    return np.sum(logps)

  def log_likelihood(self,x):
    gp2 = copy.deepcopy(self.gp)
    gp2.xc = self.xext
    gp2.xc[-1] = x
    gp2.yc = self.yext
    gp2.m = gp2._gp__fit(gp2.xc,gp2.yc,opt=False)
    gp2.m.kern.lengthscale = self.gp.m.kern.lengthscale
    gp2.m.kern.variance = self.gp.m.kern.variance
    gp2.m.Gaussian_noise.variance = self.gp.m.Gaussian_noise.variance
    return gp2.m.log_likelihood()

  def log_posterior(self,x):
    return self.log_likelihood(x) + self.log_prior(x)

  def __negative_log_posterior(self,x):
    return -self.log_posterior(x)
    
  def MAP(self):
    # Bounds which try and avoid extrapolation
    bnds = tuple((np.min(self.gp.xc[:,j]),np.max(self.gp.xc[:,j])) for j in range(self.gp.nx))
    print("Finding optimal inputs by maximising posterior. Bounds on xc are:")
    print(bnds)
    res = differential_evolution(self.__negative_log_posterior,bounds=bnds)
    resx = np.zeros(self.gp.nx)
    for i in range(self.gp.nx):
      resx[i] = self.gp.xconrevs[i].rev(res.x[i])
    self.xopt = resx
    self.xcopt = res.x

    print(f'Optimal converted model parameters are: {res.x}')
    print(f'Reverted optimal model parameters are: {resx}')
    print(f'Posterior is: {-res.fun:0.3f}')


# MCMC class inheriting from MAP
class mcmc(MAP):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)

# GP-MCMC class inheriting from mcmc
class gpmcmc(mcmc):
  def __init__(self,gp,nx_exp,nx_model,**kwargs):
    super().__init__(**kwargs)
