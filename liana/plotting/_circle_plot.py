from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal, Union
import scanpy as sc
from matplotlib import pyplot as plt
import networkx as nx
from liana.plotting._common import _prep_liana_res, _invert_scores, _filter_by, _get_top_n

from liana._docs import d
from liana._constants import Keys as K

def _pivot_liana_res(
        liana_res: pd.DataFrame,
        source_key: str = 'source',
        target_key: str = 'target',
        score_key: str = 'lr_means',
        mode: Literal['counts', 'weight'] = 'counts') -> pd.DataFrame:
    if mode not in ['counts', 'mean']:
        raise ValueError("`pivot_mode` must be 'counts' or 'mean'!")
    if mode == 'counts':
        pivot_table = liana_res.pivot_table(index=source_key, columns=target_key, aggfunc='size', fill_value=0)
    elif mode == 'mean':
        pivot_table = liana_res.pivot_table(index=source_key, columns=target_key, values=score_key, aggfunc='mean', fill_value=0)

    return pivot_table


def _scale_list(arr, min_val=1, max_val=5):
    arr = np.array(arr)
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    scaled_arr = (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val

    return scaled_arr

def _set_adata_color(adata, label, color_dict=None, hex=True):
    adata.obs[label] = adata.obs[label].astype("category")
    if color_dict:
        if not hex:
            from matplotlib.colors import to_hex

            color_dict = {x: to_hex(y) for x, y in color_dict.items()}

        _dt = _get_adata_colors(adata, label)
        _dt.update(color_dict)
        color_dict = _dt
        adata.uns[f"{label}_colors"] = [
            color_dict[x] for x in adata.obs[label].cat.categories
        ]
    else:
        if f"{label}_colors" not in adata.uns:
            sc.pl._utils._set_default_colors_for_categorical_obs(adata, label)

    return adata


def _get_adata_colors(adata, label):
    if f"{label}_colors" not in adata.uns:
        _set_adata_color(adata, label)
    return {
        x: y
        for x, y in zip(adata.obs[label].cat.categories, adata.uns[f"{label}_colors"])
    }

def get_mask_df(
        pivot_table: pd.DataFrame,
        source_cell_type: Union[list, str] = None,
        target_cell_type: Union[list, str] = None,
        mode: Literal['and', 'or'] ='or') -> pd.DataFrame:

    if source_cell_type is None and target_cell_type is None:
        return pivot_table

    if isinstance(source_cell_type, str):
        source_cell_type = [source_cell_type]
    if isinstance(target_cell_type, str):
        target_cell_type = [target_cell_type]

    mask_df = pd.DataFrame(np.zeros_like(pivot_table), index=pivot_table.index, columns=pivot_table.columns, dtype=bool)

    if mode == 'or':
        if source_cell_type is not None:
            mask_df.loc[source_cell_type] = True
        if target_cell_type is not None:
            mask_df.loc[:, target_cell_type] = True
    elif mode == 'and':
        if source_cell_type is not None and target_cell_type is not None:
            mask_df.loc[source_cell_type, target_cell_type] = True

    return pivot_table[mask_df].fillna(0)

@d.dedent
def circle_plot(
        adata: sc.AnnData,
        uns_key: Union[str, None] = K.uns_key,
        liana_res: Union[pd.DataFrame, None] = None,
        groupby: str = None,
        source_key: str = 'source',
        target_key: str = 'target',
        score_key: str = None,
        inverse_score: bool = False,
        top_n: int = None,
        orderby: str | None = None,
        orderby_ascending: bool | None = None,
        orderby_absolute: bool = False,
        filter_fun: callable = None,
        source_labels: Union[list, str] = None,
        target_labels: Union[list, str] = None,
        ligand_complex: list | str | None = None,
        receptor_complex: list | str | None = None,
        pivot_mode: Literal['counts', 'weight'] = 'counts',
        mask_mode: Literal['and', 'or'] = 'or',
        figure_size: tuple = (5, 5),
        edge_alpha: float = .5,
        edge_arrow_size: int = 10,
        edge_width_scale: tuple = (1, 5),
        node_alpha: float = 1,
        node_size_scale: tuple = (100, 400),
        node_label_offset: tuple = (0.1, -0.2),
        node_label_size: int = 8,
        node_label_alpha: float = .7,
        ):
    """
    Visualize the cell-cell communication network using a circular plot.

    Parameters
    ----------
    %(adata)s
    %(uns_key)s
    %(liana_res)s
    %(groupby)s
    %(source_key)s
    %(target_key)s
    %(score_key)s
    inverse_score : bool, optional
        Whether to invert the score, by default False. If True, the score will be -log10(score).
    %(top_n)s
    %(orderby)s
    %(orderby_ascending)s
    %(orderby_absolute)s
    %(filter_fun)s
    %(source_labels)s
    %(target_labels)s
    %(ligand_complex)s
    %(receptor_complex)s
    pivot_mode : Literal['counts', 'mean'], optional
        The mode of the pivot table, by default 'counts'.
        - 'counts': The number of connections between source and target.
        - 'mean': The mean of the values of `score_key` between source and target cell types (groupby).
    mask_mode : Literal['and', 'or'], optional
        The mode of the mask, by default 'or'.
        - 'or': Include the source or target cell type.
        - 'and': Include the source and target cell type.
    %(figure_size)s
    edge_alpha : float, optional
        The transparency of the edges, by default .5.
    edge_arrow_size : int, optional
        The size of the arrow, by default 10.
    edge_width_scale : tuple, optional
        The scale of the edge width, by default (1, 5).
    node_alpha : float, optional
        The transparency of the nodes, by default 1.
    node_size_scale : tuple, optional
        The scale of the node size, by default (100, 400).
    node_label_offset : tuple, optional
        The offset of the node label, by default (0.1, -0.2).
    node_label_size : int, optional
        The size of the node label, by default 8.
    node_label_alpha : float, optional
        The transparency of the node label, by default .7.
    """
    if groupby is None:
        raise ValueError('`groupby` must be provided!')

    liana_res = _prep_liana_res(
        adata=adata,
        source_labels=None,
        target_labels=None,
        ligand_complex=ligand_complex,
        receptor_complex=receptor_complex,
        uns_key=uns_key)

    liana_res = _filter_by(liana_res, filter_fun)
    liana_res = _get_top_n(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)

    if inverse_score:
        liana_res[score_key] = _invert_scores(liana_res[score_key])

    pivot_table = _pivot_liana_res(
        liana_res,
        source_key=source_key,
        target_key=target_key,
        score_key=score_key,
        mode=pivot_mode)

    groupby_colors = _get_adata_colors(adata, label=groupby)

    # Mask pivot table
    _pivot_table = get_mask_df(
        pivot_table,
        source_cell_type=source_labels,
        target_cell_type=target_labels,
        mode=mask_mode)

    G = nx.convert_matrix.from_pandas_adjacency(_pivot_table, create_using=nx.DiGraph())
    pos = nx.circular_layout(G)

    edge_color = [groupby_colors[cell[0]] for cell in G.edges]
    edge_width = np.asarray([G.edges[e]['weight'] for e in G.edges()])
    edge_width = _scale_list(edge_width, max_val=edge_width_scale[1], min_val=edge_width_scale[0])

    node_color = [groupby_colors[cell] for cell in G.nodes]
    node_size = pivot_table.sum(axis=1).values
    node_size = _scale_list(node_size, max_val=node_size_scale[1], min_val=node_size_scale[0])

    fig, ax = plt.subplots(figsize=figure_size)

    # Visualize network
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=edge_alpha,
        arrowsize=edge_arrow_size,
        arrowstyle='-|>',
        width=edge_width,
        edge_color=edge_color,
        connectionstyle="arc3,rad=-0.3",
        ax=ax
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=node_size,
        alpha=node_alpha,
        ax=ax
        )
    label_options = {"ec": "k", "fc": "white", "alpha": node_label_alpha}
    _ = nx.draw_networkx_labels(
        G,
        {k: v + np.array(node_label_offset) for k, v in pos.items()},
        font_size=node_label_size,
        bbox=label_options,
        ax=ax
        )

    ax.set_frame_on(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeff = 1.2
    ax.set_xlim((xlim[0] * coeff, xlim[1] * coeff))
    ax.set_ylim((ylim[0] * coeff, ylim[1]))
    ax.set_aspect('equal')

    return ax
