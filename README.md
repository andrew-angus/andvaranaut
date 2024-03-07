# Andvaranaut  

Predictive modelling and uncertainty quantification (UQ) suite. Use requires provision of target function which takes inputs and returns quantity of interest. In addition, probability distributions (from scipy.stats) for input variables should be specified if using sampling, along with optional input/output conversion methods (e.g bounded to unbounded range) for more performant machine learning surrogates.

Name backstory: https://en.wikipedia.org/wiki/Andvaranaut

To install run the following in the same directory as setup.py (can drop the --user flag if root):  

`pip3 install --user .`

## Functionality

### Current

- Latin hypercube sampling  
- UQ forward propagation  
- Parallel target function execution   
- Input \& output transformations   
- Gaussian process (GP) surrogates   
- Inverse Bayesian problems via maximum a posteriori (MAP) or Markov chain Monte Carlo (MCMC)
- GP hyperparameter optimisation via MAP or MCMC
- Hyperparameter optimisation optionally includes those necessary for complex multi-layered input and output transformations
- Custom GP mean functions
- Bayesian optimisation

### Future
 
- PCE surrogates
- Neural networks (NN)
- NN autoencoding for GP surrogate
- GP adaptive sampling  

## Tutorial

In tutorial/ is a Jupyter notebook called tutorial.ipynb. This will walk through most basic package functionality.
