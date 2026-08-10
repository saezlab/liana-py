# Installation

## Basic Installation

Install LIANA using pip:

```bash
pip install liana
```

## Conda Installation

```bash
conda install bioconda::liana
```

## Installation with Extras

LIANA offers optional dependencies for extended functionality:

### All Extras

Install all optional dependencies:

```bash
pip install 'liana[extras]'
```

This includes additional packages for:
- Multi-view analysis (decoupler, muon, mofax, mofapy2)
- Database access (omnipath)
- Differential expression (pydeseq2)
- Pathway analysis (gseapy)
- Optimization (corneto, cvxpy-base, PySCIPOpt)
- And more (cell2cell, kneed)

### Development Installation From Source

```bash
git clone https://github.com/saezlab/liana-py.git
cd liana-py
pip install -e '.[dev]'
```

## Requirements

- Python 3.10 or higher (up to 3.13)
- Core dependencies: anndata, mudata, scanpy, numba, pandas, and others are installed automatically
