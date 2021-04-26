#!/bin/python3

import GPy

class gp(surrogate):
  def __init__(self):
    self.dat = dat
    self.kern = kernel
    self.nvars = dat.x.shape[1]

  def fit(self,restarts=3,noise=False):

    kernel = 'GPy.kern.'+self.kern+f'(input_dim=)'
    if self.kern == 'RBF':
      kern = self.kern
      self.model = GPy.models.GPRegression(self.dat.x,self.dat.y,)

