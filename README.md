# Andvaranaut

https://en.wikipedia.org/wiki/Andvaranaut  

Predictive modelling and UQ suite. Use requires provision of function takes inputs and returns quantity of interest. In addition probability distributions (from scipy.stats) for input variables must be specified along with optional conversion methods (e.g bounded to unbounded range) for more efficient execution of available ML techniques.

To install run the following in the same directory as setup.py (can drop the --user flag if root):  

`pip install --user .`

## Functionality

### Current

Latin hypercube sampling
UQ forward propagation

### In Development

Gaussian process surrogates  
Adaptive sampling  
Differential evolution MCMC  
MCMC R convergence analysis  
Advanced KDE posterior plotting  

### Future

Other MCMC methods  
Marginal distributions  
PCE surrogates

## Tutorial
