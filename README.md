# Andvaranaut

https://en.wikipedia.org/wiki/Andvaranaut  

Predictive modelling and UQ suite. Use requires provision of function which takes inputs and returns quantity of interest. In addition, probability distributions (from scipy.stats) for input variables must be specified, along with optional conversion methods (e.g bounded to unbounded range) for more efficient execution of available ML techniques.

To install run the following in the same directory as setup.py (can drop the --user flag if root):  

`pip3 install --user .`

## Functionality

### Current

Latin hypercube sampling  
UQ forward propagation  
Parallel target function execution  
Gaussian process surrogates  
GP adaptive sampling  

### In Development

Differential evolution MCMC  
MCMC R convergence analysis  
Bounded KDE posterior plotting  

### Future

Other MCMC methods  
Marginal distributions  
PCE surrogates  

## Tutorial

In tutorial/ is a Jupyter notebook called tutorial.ipynb. This will walk through most basic package functionality.
