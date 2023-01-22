#!/bin/python3

from setuptools import setup,find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name="andvaranaut",
  version="0.2.0",

  author="Andrew Angus",
  author_email="andrew.angus@warwick.ac.uk",

  packages=find_packages(include=['andvaranaut','andvaranaut.*']),

  url="https://github.com/andrewanguswarwick/andvaranaut",

  description="Predictive modelling and UQ suite.",
  long_description=long_description,
  long_description_content_type="text/markdown",

  python_requires='>=3.6',
  install_requires=[
    "GPy",
    "pymc",
    "gpyopt",
    "numpy",
    "scipy",
    "py-design",
    "ray",
    "scikit-learn",
    "seaborn",
    ],
)
