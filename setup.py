#!/usr/bin/env python

from distutils.core import setup

setup(name='liana-py',
      version='0.1',
      description='LIANA - a LIgand-receptor ANalysis frAmework',
      author='Daniel Dimitrov',
      author_email='daniel.dimitrov@uni-heidelberg.de',
      url='https://github.com/saezlab/liana-py',
      packages=['liana'],
      install_requires=["numba",
                        "tqdm",
                        "pandas",
                        "anndata",
                        "scanpy"],
      python_requires=">=3.6",
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent"]
      )
