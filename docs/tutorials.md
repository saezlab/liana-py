# Tutorials

Every tutorial is a runnable notebook that starts from a public dataset loaded via {mod}`liana.ds`.

## Which analysis do I need?

Use the tree below to find a starting point.
Broad, data-driven choices sit at the top and trickle down to specific methods; click a node to open its tutorial.

```{include} ../README.md
:start-after: <!-- decision-tree-start -->
:end-before: <!-- decision-tree-end -->
```

The tree is a guide rather than an exhaustive map: the methods are modular and can be adapted or combined across data types and questions.

## Getting started

The basics of LIANA+: how a method is called, and where the prior knowledge it scores comes from.

```{toctree}
:maxdepth: 1

tutorials/notebooks/basic_usage.ipynb
tutorials/notebooks/prior_knowledge.ipynb
```

## Dissociated single-cell data

Inference within a single sample, and across samples or conditions.

```{toctree}
:maxdepth: 1

tutorials/notebooks/targeted.ipynb
tutorials/notebooks/liana_pyCrossTalkeR.ipynb
tutorials/notebooks/mofatalk.ipynb
tutorials/notebooks/mofacellular.ipynb
tutorials/notebooks/liana_c2c.ipynb
```

## Spatially-resolved data

Interactions that are constrained, scored, or learnt from spatial coordinates.

```{toctree}
:maxdepth: 1

tutorials/notebooks/bivariate.ipynb
tutorials/notebooks/misty.ipynb
tutorials/notebooks/inflow_score.ipynb
tutorials/notebooks/inflow_mofaflex.ipynb
tutorials/notebooks/LRIC_tutorial.ipynb
```

## Multi-modal data

Interactions between modalities, such as transcriptome and surface protein or metabolite.

```{toctree}
:maxdepth: 1

tutorials/notebooks/sc_multi.ipynb
tutorials/notebooks/sma.ipynb
```
