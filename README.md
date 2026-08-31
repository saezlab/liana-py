# LIANA+: an all-in-one cell-cell communication framework <img src="https://raw.githubusercontent.com/scverse/liana-py/main/docs/_static/logo.png" align="right" height="125">

<!-- badges: start -->
[![main](https://github.com/scverse/liana-py/actions/workflows/test.yml/badge.svg)](https://github.com/scverse/liana-py/actions)
[![GitHub issues](https://img.shields.io/github/issues/scverse/liana-py.svg)](https://github.com/scverse/liana-py/issues/)
[![Documentation Status](https://readthedocs.org/projects/liana-py/badge/?version=latest)](https://liana-py.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/scverse/liana-py/graph/badge.svg?token=2HvdhecFQU)](https://codecov.io/gh/scverse/liana-py)
[![Downloads](https://static.pepy.tech/badge/liana)](https://pepy.tech/project/liana)
<!-- badges: end -->

LIANA+ is a scalable framework that adapts and extends existing methods and knowledge to study cell-cell communication in single-cell, spatially-resolved, and multi-modal omics data. It is part of the [scverse ecosystem](https://github.com/scverse), and relies on [AnnData](https://github.com/scverse/anndata) & [MuData](https://github.com/scverse/mudata) objects as input.

<img src="https://raw.githubusercontent.com/scverse/liana-py/main/docs/_static/abstract.png" width="700" align="center">

## Contributions

We welcome suggestions, ideas, and contributions! Please do not hesitate to contact us, open issues, and check the [contributions guide](https://liana-py.readthedocs.io/en/latest/contributing.html).

## Vignettes
A set of extensive vignettes can be found in the [LIANA+ documentation](https://liana-py.readthedocs.io/en/latest/).

## Decision Tree

Use the tree below to find a starting point for your analysis. Broad, data-driven choices sit at the top and trickle down to specific methods (click a node to open its tutorial).

```mermaid
flowchart TD
    Start{What is your data?}

    %% ===== Spatially-resolved =====
    Start -->|Spatially-resolved| Res{Resolution?}
    Res -->|Single-cell| ScType{Analysis type?}
    ScType -->|Interaction scoring| Inflow[Inflow Score]
    ScType -->|Interaction scoring| ScConstr[Standard LR methods<br/>spatially-constrained]
    ScType -->|Spatial co-occurrence| LRIC[LRIC]
    ScType -->|Unsupervised| InflowMofa[Communication Programs<br/>Inflow + MOFA-Flex]
    Res -->|Spot-based| SpType{Analysis type?}
    SpType -->|Bivariate| LocalQ{Local<br/>interactions?}
    LocalQ -->|Yes| Local[Local Bivariate Metrics]
    LocalQ -->|No| Global[Global Bivariate Metrics]
    SpType -->|Unsupervised| MISTy[Multi-view Learning<br/>MISTy]

    %% ===== Dissociated single-cell =====
    Start -->|Dissociated single-cell| Compare{Compare across<br/>samples?}
    Compare -->|No| Steady[Steady-state LR Inference]
    Compare -->|Yes| Contrast{Specific<br/>contrast?}
    Contrast -->|Yes| Targeted[Targeted Differential]
    Contrast -->|Yes| CrossTalk[pyCrossTalkeR<br/>network differential]
    Contrast -->|No| MOFA[MOFA+]
    Contrast -->|No| Tensor[Tensor-cell2cell]
    Tensor --> TensorExt[Extended Tutorials<br/>ccc-protocols]
    Tensor --> TensorMet[CCC Patterns: protein + metabolite<br/>Tensor-cell2cell CTCA]

    %% ===== Multi-modal =====
    Start -->|Multi-modal| ModalSp{Spatial?}
    ModalSp -->|Yes| SMA[Multi-Modal Spatial]
    ModalSp -->|No| SCMulti[Multi-Modal Single-Cell]
    SCMulti --> Metab[Metabolite-mediated CCC]

    %% ===== Styling =====
    classDef decision fill:#f5f5f5,stroke:#37474f,stroke-width:1px,color:#263238;
    classDef spatial fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef dissoc fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef multimodal fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c;
    classDef external fill:#ffffff,stroke:#9e9e9e,stroke-dasharray:5 3,color:#424242;

    class Start,Res,ScType,SpType,LocalQ,Compare,Contrast,ModalSp decision;
    class Inflow,LRIC,ScConstr,InflowMofa,Local,Global,MISTy spatial;
    class Steady,Targeted,CrossTalk,MOFA,Tensor dissoc;
    class SMA,SCMulti,Metab multimodal;
    class TensorExt,TensorMet external;

    %% ===== Links (click events) =====
    click Inflow "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/inflow_score.html"
    click LRIC "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/LRIC_tutorial.html"
    click InflowMofa "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/inflow_mofaflex.html"
    click ScConstr "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/inflow_score.html"
    click Local "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/bivariate.html"
    click Global "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/bivariate.html"
    click MISTy "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/misty.html"
    click Steady "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/basic_usage.html"
    click Targeted "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/targeted.html"
    click CrossTalk "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/liana_pyCrossTalkeR.html"
    click MOFA "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/mofatalk.html"
    click Tensor "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/liana_c2c.html"
    click TensorExt "https://ccc-protocols.readthedocs.io/en/latest/"
    click TensorMet "https://earmingol.github.io/cell2cell/tutorials/Version2/Tensor-cell2cell-CTCA-LIANA/"
    click SMA "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/sma.html"
    click SCMulti "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/sc_multi.html"
    click Metab "https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/sc_multi.html#metabolite-mediated-ccc-from-transcriptomics-data"
```

This tree is a guide rather than an exhaustive map: the methods are modular and can be adapted or combined across data types and questions, and all of them typically build on curated prior knowledge (see the [prior knowledge](https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/prior_knowledge.html) tutorial for working with ligand–receptor and other resources).

## API
For further information please check LIANA's [API documentation](https://liana-py.readthedocs.io/en/latest/api.html).

## Cite LIANA+:

```
@article {Dimitrov2024,
    author = {Dimitrov, Daniel and Sch{\"a}fer, Philipp Sven Lars and  Farr, Elias and Rodriguez-Mier, Pablo and Lobentanzer, Sebastian and Badia-i-Mompel, Pau and Dugourd, Aurelien and Tanevski, Jovan and Ramirez Flores, Ricardo Omar and Saez-Rodriguez, Julio},
    title = {LIANA+ provides an all-in-one framework for cell--cell communication inference},
    journal = {Nature Cell Biology},
    year = {2024},
    volume = {26},
    number = {9},
    pages = {1613--1622},
    DOI = {10.1038/s41556-024-01469-w},
    URL = {https://doi.org/10.1038/s41556-024-01469-w}
}
```

```
@article {Dimitrov2022,
    author = {Dimitrov, Daniel and T{\"u}rei, D{\'e}nes and Garrido-Rodriguez, Martin and Burmedi, Paul L. and Nagai, James S. and Boys, Charlotte and Ramirez Flores, Ricardo O. and Kim, Hyojin and Szalai, Bence and Costa, Ivan G. and Valdeolivas, Alberto and Dugourd, Aur{\'e}lien and Saez-Rodriguez, Julio},
    title = {Comparison of methods and resources for cell-cell communication inference from single-cell RNA-Seq data},
    journal = {Nature Communications},
    year = {2022},
    volume = {13},
    number = {1},
    pages = {3224},
    DOI = {10.1038/s41467-022-30755-0},
    URL = {https://doi.org/10.1038/s41467-022-30755-0}
}
```

Please also consider citing any of the methods and/or resources that were particularly relevant for your research!

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/liana-py/issues
[tests]: https://github.com/scverse/liana-py/actions/workflows/test.yml
[documentation]: https://liana-py.readthedocs.io
[changelog]: https://liana-py.readthedocs.io/en/latest/changelog.html
[api documentation]: https://liana-py.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/liana
