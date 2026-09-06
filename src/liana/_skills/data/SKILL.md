---
name: liana
description: Cell-cell communication (CCC) inference with the liana Python package (LIANA+, scverse). Use for any task involving liana or ligand-receptor (LR) analysis of AnnData/MuData objects. Triggers on steady-state LR scoring (rank_aggregate, CellPhoneDB, CellChat, NATMI, Connectome, SingleCellSignalR, logFC, scSeqComm); multi-sample or differential CCC (by_sample, MOFA+, Tensor-cell2cell, df_to_lr, pyCrossTalkeR); spatial CCC on Visium, Xenium, MERFISH, CosMx or slide-seq (spatial_neighbors, bivariate local/global metrics, Moran's R, Inflow, LRIC, cross-PCF, MISTy); multimodal CITE-seq or spatial metabolomics; metabolite-mediated CCC via MetalinksDB; LR resources and orthology for mouse or other organisms (consensus, mouseconsensus, OmniPath, HCOP); liana plots (dotplot, tileplot, circle_plot). This is a router skill - read the matching file under references/ before writing liana code, because output locations and several defaults are non-obvious.
---

# liana (LIANA+)

`import liana as li`. Submodules: `li.mt` methods, `li.pp` preprocessing (spatial graphs,
transforms), `li.rs` prior knowledge, `li.pl` plots, `li.ms` multi-sample helpers, `li.ds` datasets.
Every method accepts an `AnnData` or a `MuData`.

## Workflow

1. **Intake.** Ask for the object or a path to it, and say explicitly that a description of the
   data is fine instead if it cannot be shared (patient data, privacy). If a path is given, run
   the snippet below and infer what you can. Ask only what the object cannot tell you:
   the analysis aim, and which `obs` column holds the condition and cell type if several matching columns exist.
   Without an object, ask all of: resolution (dissociated cells / spots / single cells with
   coordinates); coordinate units (µm or pixels); one or several samples, conditions; cell-type
   labels (an `obs` column, or proportions in `obsm`); modalities beyond RNA (protein, measured
   metabolites, chromatin); organism; whether `.X` is
   log-normalised; and the aim.
2. **Select** the branch from the table.
3. **Read** that one reference file. Also read `prior-knowledge.md` for non-human data or a custom
   LR list, and `outputs-and-plotting.md` before plotting or interpreting `liana_res`.
4. **Run**, say which method was used and why.

```python
import anndata as ad, mudata as mu, numpy as np
a = mu.read(path) if path.endswith(".h5mu") else ad.read_h5ad(path)
print(type(a).__name__, a.shape, "mods:", list(getattr(a, "mod", {})))
for c in a.obs.columns[:40]:
    u = a.obs[c].unique(); print(c, len(u), list(u[:6]))
print("obsm:", list(a.obsm), "obsp:", list(a.obsp), "layers:", list(a.layers), "raw:", a.raw is not None)
X = a.X[:200]; X = X.toarray() if hasattr(X, "toarray") else X
print("X min/max:", X.min(), X.max(), "integer-like:", np.allclose(X, X.round()), "var sample:", list(a.var_names[:5]))
```

## Selection

| Data and aim | Read |
|---|---|
| Dissociated single cells, one dataset: rank LR interactions between cell types | [single-cell-lr.md](references/single-cell-lr.md) |
| Dissociated single cells, several samples or conditions: what changes between them | [multisample.md](references/multisample.md) |
| Spatial spots (Visium) or cells: where do ligand and receptor co-vary in space, local and global scores | [spatial-bivariate.md](references/spatial-bivariate.md) |
| Spatial single cells with cell-type labels (Xenium, MERFISH, CosMx): which types signal to which via which LRs | [inflow.md](references/inflow.md) |
| Spatial single cells: at what distance do interactions occur, spatial scale of cell-type pairs | [lric.md](references/lric.md) |
| Spatial, unsupervised: what in a spot's neighbourhood predicts its expression, multi-view learning | [misty.md](references/misty.md) |
| Metabolite-mediated CCC, estimated from RNA or measured (MSI) | [metabolites.md](references/metabolites.md) |
| Resources, organism translation, custom LR lists, gene sets from LRs | [prior-knowledge.md](references/prior-knowledge.md) |
| Reading `liana_res`, score semantics, plotting | [outputs-and-plotting.md](references/outputs-and-plotting.md) |

The README decision tree is a starting point, not a rule: methods are modular and combine.
Multi-modal input is not a branch. Every method takes a `MuData`; each file has a Variants section
saying how. Each file ends with the optional features to surface only when the data calls for them.

**Widen the question.** Every branch also covers non-protein mediators. Any RNA dataset, dissociated or
spatial, can be scored for metabolite-mediated CCC by estimating metabolite abundance from enzyme and
transporter expression (MetalinksDB), and a second modality (protein, measured metabolites) can
supply the ligands or the receptors. Offer this whenever the user wants the complete picture, names
metabolites, neurotransmitters, hormones or lipids, or has such a modality: read `metabolites.md`.

## Facts that apply everywhere

- **Input**: (typically) non-negative, library-size normalised, log1p expression in `.X` (or `layer=`).
  `use_raw` defaults to `False`. Scaled or z-scored values give NaN `logfc` and NaN
  `specificity_rank`, and negative values break the single-cell methods. Raw counts only trigger a warning.
  Always true for methods under liana.mt.sc, liana.mt.sp method can make exceptions to those.
- **"Please check if appropriate organism/ID type was provided!"** means the resource and
  `var_names` do not overlap. Tell the user both causes: `var_names` that are not gene symbols
  (Ensembl IDs, the wrong matrix), and non-human data with the human `consensus` resource. For
  mouse use `resource_name="mouseconsensus"`; for a fuller map or any other organism translate
  the resource with HCOP orthologs (`li.rs.get_hcop_orthologs`, read `prior-knowledge.md`).
- Default resource `consensus` uses **human gene symbols**.
- **Complexes**: subunits joined by `_`. `ligand` / `receptor` columns hold the least-expressed
  subunit; `ligand_complex` / `receptor_complex` hold the full name.
- **Where results land**: single-cell methods write `adata.uns["liana_res"]` in place;
  `bivariate` and `inflow` return a **new** AnnData; `lric`, `cross_pcf` and MISTy write `.uns` keys.
- **Two thresholds**, both 0.05 by default and worth tuning: `expr_prop` (single-cell methods) is
  the fraction of cells within a cell-type group expressing a gene; `nz_prop` (spatial methods only)
  is the fraction of all cells or spots with a non-zero value.
- Plot with `li.pl.*` (plotnine, returns a `ggplot`). Do not rebuild these plots from matplotlib primitives.
- Extras: `pip install 'liana[extras]'` covers MOFA, Tensor-cell2cell, pseudobulk DE, causal
  networks and MetalinksDB. Downloads (`li.ds.kang_2018`, HCOP tables, MetalinksDB) go to the cwd.

## Citing

Always cite LIANA+ (Dimitrov et al., Nat Cell Biol 2024, doi:10.1038/s41556-024-01469-w) plus the
original paper of the method and resource used: `li.mt.<method>.reference` holds each single-cell
method's citation, and `li.rs.show_resources()` names the resource databases. For the consensus resource
and rank aggregate also cite Dimitrov et al., Nat Commun 2022 (doi:10.1038/s41467-022-30755-0). Inflow and
LRIC are unpublished (Alsayah et al., in preparation): cite LIANA+ for them meanwhile.
