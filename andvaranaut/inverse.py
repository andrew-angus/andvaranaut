#!/bin/python3

from andvaranaut.utils import _core
from andvaranaut.forward import GP
import numpy as np
from scipy.optimize import differential_evolution
import scipy.stats as st
import copy

# Maximum a posteriori class
class MAP(_core):
  def __init__(self,nx_exp,nx_model,**kwargs):
    # Check inputs
    if (not isinstance(nx_exp,int)) or (nx_exp < 0):
      raise Exception(\
          'Error: must specify an integer number of experimental input parameters => 0') 
    if (not isinstance(nx_model,int)) or (nx_model < 1):
      raise Exception('Error: must specify an integer number of model input parameters > 0') 
    # Allow for only specifying priors for model parameters
    if (isinstance(kwargs['priors'],list)) and (len(kwargs['priors']) == nx_model):
      for i in range(nx_exp):
        kwargs['priors'].insert(0,st.norm())
    super().__init__(nx=nx_exp+nx_model,**kwargs)
    # Initialise new attributes
    self.nx_exp = nx_exp # Observable input dimensions
    self.nx_model = nx_model # Model input dimensions
    self.x_obv = None
    self.y_obv = None
    self.y_noise = None
    self.obvs = None
    self.x_opt = None

  # Set your experimental x and y with noise
  def set_observations(self,y,y_noise=None,x_exp=None):
    # Check y input
    if not isinstance(y,np.ndarray) or len(y.shape) != 2 \
        or y.dtype != 'float64' or y.shape[1] != self.ny:
      raise Exception(\
          "Error: Setting data requires a 2d numpy array of float64 outputs")
    # Check/modify other inputs
    self.obvs = len(y)
    if y_noise is None:
      y_noise = np.ones((self.obvs,self.ny))*np.finfo(np.float64).eps
    else:
      if not isinstance(y_noise,np.ndarray) or len(y_noise.shape) != 2 \
          or y_noise.dtype != 'float64' or y_noise.shape[1] != self.ny:
        raise Exception(\
            "Error: Setting data requires a 2d numpy array of float64 noises"+\
            " of the same shape as y")
    if x_exp is None:
      if self.nx_exp != 0:
        raise Exception("Error: must provide x_exp values for each observation")
    else:
      if not isinstance(x_exp,np.ndarray) or len(x_exp.shape) != 2 \
          or x_exp.dtype != 'float64' or x_exp.shape[1] != self.nx_exp:
        raise Exception(\
            "Error: Setting data requires a 2d numpy array of float64 exp inputs"+\
            " of shape (len(y),nx_exp)")
    # Set obvs
    self.x_obv = np.zeros((self.obvs,self.nx))
    self.x_obv[:,:self.nx_exp] = x_exp
    self.y_obv = y
    self.y_noise = y_noise

  # Log prior method acting on specified model inputs
  def log_prior(self,x):
    logps = np.zeros(self.nx_model)
    for i in range(self.nx_model):
      logps[i] = self.priors[i+self.nx_exp].logpdf(x[i])
    return np.sum(logps)

  # Gaussian log likelihood method acting on specified model inputs
  def log_likelihood(self,x):
    self.x_obv[:,self.nx_exp:] = x
    xsamps,fvals = self._core__vector_solver(self.x_obv,verbose=False)
    if len(xsamps) != self.obvs:
      raise Exception("Error: one or more function evaluations failed to return valid result.")
    res = -self.obvs*self.ny*0.5*np.log(2*np.pi)
    res -= np.sum(np.log(self.y_noise))
    res -= np.sum(0.5/np.power(self.y_noise,2)*np.power(fvals-self.y_obv,2))
    return res

  # Log posterior method acting on specified model inputs
  def log_posterior(self,x):
    return self.log_likelihood(x) + self.log_prior(x)

  # Negative version of log posterior for minimizing using standard libraries
  def __negative_log_posterior(self,x):
    return -self.log_posterior(x)
    
  def opt(self):
    # Bounds which try and avoid extrapolation
    bnd = 0.999999999999999
    bnds = tuple(self.priors[i+self.nx_exp].interval(bnd) for i in range(self.nx_model))
    print("Finding optimal inputs by maximising posterior. Bounds on x are:")
    print(bnds)
    res = differential_evolution(self.__negative_log_posterior,bounds=bnds)
    self.xopt = res.x

    print(f'Optimal converted model parameters are: {res.x}')
    print(f'Posterior is: {-res.fun:0.3f}')

# Quick and dirty maximum a posteriori class using a GP
class gpmap:
  def __init__(self,nx_exp,nx_model,gp):
    # Check inputs
    if (not isinstance(nx_exp,int)) or (nx_exp < 0):
      raise Exception('Error: must specify an integer number of model input parameters => 0') 
    if (not isinstance(nx_model,int)) or (nx_model < 1):
      raise Exception('Error: must specify an integer number of model input parameters > 0') 
    if (not isinstance(gp,GP)):
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
