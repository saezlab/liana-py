# LIANA+: an all-in-one cell-cell communication framework <img src="https://raw.githubusercontent.com/saezlab/liana-py/dev/docs/source/_static/logo.png?raw=true" align="right" height="125">

<!-- badges: start -->
[![main](https://github.com/saezlab/liana-py/actions/workflows/main.yml/badge.svg)](https://github.com/saezlab/liana-py/actions)
[![GitHub issues](https://img.shields.io/github/issues/saezlab/liana-py.svg)](https://github.com/saezlab/liana-py/issues/)
[![Documentation Status](https://readthedocs.org/projects/liana-py/badge/?version=latest)](https://liana-py.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/saezlab/liana-py/branch/main/graph/badge.svg?token=TM0P29KKN5)](https://codecov.io/gh/saezlab/liana-py)
[![Downloads](https://static.pepy.tech/badge/liana)](https://pepy.tech/project/liana)
<!-- badges: end -->

LIANA+ is a scalable framework that integrates and extends existing methods and knowledge to study cell-cell communication in single-cell, spatially-resolved, and multi-modal omics data. It works in conjunction with the [scverse ecosystem](https://github.com/scverse), and it relies on [AnnData](https://github.com/scverse/anndata) & [MuData](https://github.com/scverse/mudata) objects as input.

<img src="https://raw.githubusercontent.com/saezlab/liana-py/main/docs/source/_static/abstract.png" width="700" align="center">

## Development & Contributions

LIANA+ is still under development. Nevertheless, the API visible to the user should be stable, specifically `liana.method.sc`.

We welcome suggestions, ideas, and contributions! Please use do not hesitate to contact us, or use the issues or the [LIANA+ Development project](https://github.com/orgs/saezlab/projects/16) to make suggestions.

## Tutorials

### Single-cell/Dissociated Data

- [LIANA's basic tutorial](https://liana-py.readthedocs.io/en/latest/notebooks/basic_usage.html) in dissociated single-cell data

### Spatial Data

- [Learn spatially-informed relationships with MISTy](https://liana-py.readthedocs.io/en/latest/notebooks/misty.html) across (multi-) views.

- [Estimate local spatially-informed bivariate metrics](https://liana-py.readthedocs.io/en/latest/notebooks/bivariate.html). This tutorial shows how to estimate local spatially-informed bivariate metrics, such as the spatially-informed Pearson correlation coefficient or Cosine similarity.

### Multi-condition

- [Hypothesis-testing for CCC with PyDeSeq2](https://liana-py.readthedocs.io/en/latest/notebooks/targeted.html) that also shows the inference of causal **intracellular** signalling networks, downstream of CCC events.

- [Multicellular programmes with MOFA](https://liana-py.readthedocs.io/en/latest/notebooks/mofacellular.html). Using MOFA to obtain coordinates gene expression programmes across samples and conditions, as done in [Ramirez et al., 2023](https://europepmc.org/article/ppr/ppr620471).

- [LIANA with MOFA](https://liana-py.readthedocs.io/en/latest/notebooks/mofatalk.html). Using MOFA to infer intercellular communication programmes across samples and conditions, as initially proposed by Tensor-cell2cell.

- [LIANA with Tensor-cell2cell](https://liana-py.readthedocs.io/en/latest/notebooks/liana_c2c.html) to extract intercellular communication programmes across samples and conditions. Extensive tutorials combining LIANA & [Tensor-cell2cell](https://www.nature.com/articles/s41467-022-31369-2) are available [here](https://ccc-protocols.readthedocs.io/en/latest/index.html).


### Others

- We also refer users to the [Cell-cell communication chapter](https://www.sc-best-practices.org/mechanisms/cell_cell_communication.html) in the [best-practices guide from Theis lab](https://www.nature.com/articles/s41576-023-00586-w). There we provide an overview of the common limitations and assumptions in CCC inference from (dissociated single-cell) transcriptomics data.


## API
For further information please check LIANA's [API documentation](https://liana-py.readthedocs.io/en/latest/api.html).


## Cite LIANA+:

Dimitrov D., Schäfer P.S.L, Farr E., Rodriguez Mier P., Lobentanzer S., Dugourd A., Tanevski J., Ramirez Flores R.O. and Saez-Rodriguez J. 2023 LIANA+: an all-in-one cell-cell communication framework. BioRxiv. https://www.biorxiv.org/content/10.1101/2023.08.19.553863v1

Dimitrov, D., Türei, D., Garrido-Rodriguez M., Burmedi P.L., Nagai, J.S., Boys, C., Flores, R.O.R., Kim, H., Szalai, B., Costa, I.G., Valdeolivas, A., Dugourd, A. and Saez-Rodriguez, J. Comparison of methods and resources for cell-cell communication inference from single-cell RNA-Seq data. Nat Commun 13, 3224 (2022). https://doi.org/10.1038/s41467-022-30755-0

Similarly, please consider citing any of the methods and/or resources implemented in liana, that were particularly relevant for your research!
