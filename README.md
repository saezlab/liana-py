# LIANA+: a one-stop-shop framework for cell-cell communication <img src="docs/source/_static/logo.png?raw=true" align="right" height="125">

<!-- badges: start -->
[![main](https://github.com/saezlab/liana-py/actions/workflows/main.yml/badge.svg)](https://github.com/saezlab/liana-py/actions)
[![GitHub issues](https://img.shields.io/github/issues/saezlab/liana-py.svg)](https://github.com/saezlab/liana-py/issues/)
[![Documentation Status](https://readthedocs.org/projects/liana-py/badge/?version=latest)](https://liana-py.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/saezlab/liana-py/branch/main/graph/badge.svg?token=TM0P29KKN5)](https://codecov.io/gh/saezlab/liana-py)
[![Downloads](https://pepy.tech/badge/liana)](https://pepy.tech/project/liana)
<!-- badges: end -->

LIANA+ is an efficient framework that integrates and extends existing methods and knowledge to study cell-cell communication in single-cell, spatially-resolved, and multi-modal omics data. 

<img src="abstract.png" width="500">


## Tutorials

### Single-cell/Dissociated Data

- [LIANA's basic tutorial](https://liana-py.readthedocs.io/en/latest/notebooks/basic_usage.html) in dissociated single-cell data

### Spatial Data

- [Learn spatially-informed relationships  with MISTy](https://liana-py.readthedocs.io/en/latest/notebooks/misty.html) across (multi-)views

- Estimate local spatially-informed bivariate metrics with [LIANA's bivariate scores](https://liana-py.readthedocs.io/en/latest/notebooks/bivariate.html)

### Multi-condition

- [Hypothesis-testing for CCC with PyDeSeq2](https://liana-py.readthedocs.io/en/latest/notebooks/targeted.html) that also shows the inference of causal **intracellular** signalling networks

- [Multicellular programmes with MOFA](https://liana-py.readthedocs.io/en/latest/notebooks/mofacellular.html). Using MOFA to obtain coordinates
gene expression programmes across samples and conditions, as done in [Ramirez et al., 2023](https://europepmc.org/article/ppr/ppr620471)

- [LIANA with MOFA](https://liana-py.readthedocs.io/en/latest/notebooks/mofatalk.html). Using MOFA to infer intercellular communication programmes across samples and conditions, as initially proposed by cell2cell-Tensor

- [LIANA with cell2cell-Tensor](https://liana-py.readthedocs.io/en/latest/notebooks/liana_c2c.html) to extract intercellular communication programmes across samples and conditions. Extensive tutorials combining LIANA & [cell2cell-Tensor](https://www.nature.com/articles/s41467-022-31369-2) are available [here](https://ccc-protocols.readthedocs.io/en/latest/index.html).


### Others

- We also refer users to the [Cell-cell communication chapter](https://www.sc-best-practices.org/mechanisms/cell_cell_communication.html) in the [best-practices guide from Theis lab](https://www.nature.com/articles/s41576-023-00586-w). There we provide an overview of the common limitations and assumptions in CCC inference from (dissociated single-cell) transcriptomics data.

- A set of in-depth tutorials [combining LIANA and Tensor-cell2cell](https://ccc-protocols.readthedocs.io/en/latest/), where we describe how to obtain intercellular communication programmes across samples and conditions in detail.


## API
For further information please check LIANA's [API documentation](https://liana-py.readthedocs.io/en/latest/api.html).



## Cite LIANA+:

Dimitrov, D. ... Saez-Rodriguez, J.

Similarly, please consider citing any of the methods and/or resources implemented in liana, that were particularly relevant for your research!
