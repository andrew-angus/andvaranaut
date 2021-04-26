#!/bin/python3

import GPy

class gp():
  def __init__(self,dat,kernel='Matern32'):
    self.dat = dat
    self.kern = kernel
    self.nvars = dat.x.shape[1]

  def fit(self,restarts=3,noise=False):

    kernel = 'GPy.kern.'+self.kern+f'(input_dim=)'
    if self.kern == 'RBF':
      kern = self.kern
      self.model = GPy.models.GPRegression(self.dat.x,self.dat.y,)

