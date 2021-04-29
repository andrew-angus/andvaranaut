#!/bin/python3

import ray
import multiprocessing as mp
import os
import numpy as np


# Function which wraps serial function for executing in parallel directories
@ray.remote
def __parallel_wrap(inp,idx,func):
  d = f'./parallel/task{idx}'
  os.system(f'mkdir {d}')
  os.chdir(d)
  res = func(inp)
  os.chdir('../..')
  return res

# Method which takes function, and 2D array of inputs
# Then runs in parallel for each set of inputs
# Returning 2D array of outputs
def parallel_runs(func,inps,nps):
    
  # Ensure number of requested processors is reasonable
  assert (nps <= mp.cpu_count()),\
      "Error: number of processors selected exceeds available."
  
  # Create parallel directory for tasks
  os.system('mkdir parallel')

  # Run function in parallel    
  ray.init(num_cpus=nps,log_to_driver=False,ignore_reinit_error=True)
  l = len(inps)
  ids = []
  for i in range(len(inps)):
    ids += [__parallel_wrap.remote(inps[i],i,func)]
  outs = []; fail = -1
  for i in range(len(inps)):
    try:
      outs.append(ray.get(ids[i]))
    except:
      fail = i
      print(f"Warning: parallel run {i} failed.",\
        "Check number of inputs/outputs and whether input ranges are valid.",\
        "Will save previous succesful runs to database.")
      ray.shutdown()
      break
  ray.shutdown()
  
  # Reshape outputs to 2D array
  if isinstance(outs[0],np.ndarray):
    outs = np.array(outs)
  else:
    outs = np.array(outs).reshape((l,1))

  return outs, fail

