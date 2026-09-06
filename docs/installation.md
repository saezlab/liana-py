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
git clone https://github.com/scverse/liana-py.git
cd liana-py
pip install -e '.[dev]'
```

## Claude Code Skill

LIANA+ bundles an Agent Skill for Claude Code. After installing liana, run `liana-install-skills` once to copy it to `~/.claude/skills/liana/`; Claude Code picks it up automatically in every project (type `/skills` to confirm) and consults it for liana tasks. Refresh the copy with `liana-install-skills --force` after upgrading liana.

## Requirements

- Python 3.12 or higher
- Core dependencies: anndata, mudata, scanpy, numba, pandas, and others are installed automatically
