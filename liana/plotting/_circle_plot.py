import pandas as pd
import numpy as np
from typing import Literal, Union
import scanpy as sc
from matplotlib import pyplot as plt
from liana._logging import _logg


def pivot_liana_res(
        liana_res: pd.DataFrame, 
        source: str = 'source', 
        target: str = 'target',
        values: str = 'lr_means',
        mode: Literal['counts', 'weight'] = 'counts') -> pd.DataFrame:
    """
    Pivot the liana_res DataFrame to a table with source and target as index and columns, respectively.

    Parameters
    ----------
    liana_res : pd.DataFrame
        The DataFrame containing the results of the liana analysis.
    source : str, optional
        The column name of the source cell type, by default 'source'.
    target : str, optional
        The column name of the target cell type, by default 'target'.
    values : str, optional
        The column name of the values to be aggregated, by default 'lr_means'.
        Only used when mode='weight'.
    mode : Literal['counts', 'weight'], optional
        The mode of the pivot table, by default 'counts'.
        - 'counts': The number of connections between source and target.
        - 'weight': The mean of the values between source and target.
    """
    if mode == 'counts':
        pivot_table = liana_res.pivot_table(index=source, columns=target, aggfunc='size', fill_value=0)
    elif mode == 'weight':
        pivot_table = liana_res.pivot_table(index=source, columns=target, values=values, aggfunc='mean', fill_value=0)

    return pivot_table


def scale_list(arr, min_val=1, max_val=5):
    arr = np.array(arr)
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    scaled_arr = (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
    
    return scaled_arr

def set_adata_color(adata, label, color_dict=None, hex=True):
    adata.obs[label] = adata.obs[label].astype("category")
    if color_dict:
        if not hex:
            from matplotlib.colors import to_hex
            
            color_dict = {x: to_hex(y) for x, y in color_dict.items()}

        _dt = get_adata_color(adata, label)
        _dt.update(color_dict)
        color_dict = _dt
        adata.uns[f"{label}_colors"] = [
            color_dict[x] for x in adata.obs[label].cat.categories
        ]
    else:
        if f"{label}_colors" not in adata.uns:
            sc.pl._utils._set_default_colors_for_categorical_obs(adata, label)

    return adata


def get_adata_color(adata, label):
    if f"{label}_colors" not in adata.uns:
        set_adata_color(adata, label)
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


def circle_plot(
        adata,
        liana_res: Union[pd.DataFrame, str],
        pivot_mode: Literal['counts', 'weight'] = 'counts',
        source_key: str = 'source',
        target_key: str = 'target',
        values_key: str = 'lr_means',
        cell_type_key: str = 'cell_type',
        source_cell_type: Union[list, str] = None,
        target_cell_type: Union[list, str] = None,
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
    adata : AnnData
        The AnnData object.
    liana_res : Union[pd.DataFrame, str]
        The results of the liana analysis. If a string is provided, it will be used as the key to retrieve the results from adata.uns.
    pivot_mode : Literal['counts', 'weight'], optional
        The mode of the pivot table, by default 'counts'.
        - 'counts': The number of connections between source and target.
        - 'weight': The mean of the values between source and target.
    source_key : str, optional
        The column name of the source cell type, by default 'source'.
    target_key : str, optional
        The column name of the target cell type, by default 'target'.
    values_key : str, optional
        The column name of the values to be aggregated, by default 'lr_means'.
        Only used when pivot_mode='weight'.
    cell_type_key : str, optional
        The column name of the cell type, by default 'cell_type'.
    source_cell_type : Union[list, str], optional
        The source cell type to be included in the plot, by default None.
    target_cell_type : Union[list, str], optional
        The target cell type to be included in the plot, by default None.
    mask_mode : Literal['and', 'or'], optional
        The mode of the mask, by default 'or'.
        - 'or': Include the source or target cell type.
        - 'and': Include the source and target cell type.
    figure_size : tuple, optional
        The size of the figure, by default (5, 5).
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
    
    try:
        import networkx as nx
    except ImportError:
        _logg('Please install networkx to use this function.')
        return
    
    if isinstance(liana_res, str):
        if liana_res in adata.uns.keys():
            liana_res = adata.uns[liana_res]
        else:
            raise KeyError(f'{liana_res} not found in adata.uns')
    
    pivot_table = pivot_liana_res(
        liana_res, 
        source=source_key, 
        target=target_key, 
        values=values_key, 
        mode=pivot_mode)

    cell_type_colors = get_adata_color(adata, label=cell_type_key)

    # Mask pivot table
    _pivot_table = get_mask_df(
        pivot_table, 
        source_cell_type=source_cell_type, 
        target_cell_type=target_cell_type, 
        mode=mask_mode)

    G = nx.convert_matrix.from_pandas_adjacency(_pivot_table, create_using=nx.DiGraph())
    pos = nx.circular_layout(G)

    edge_color = [cell_type_colors[cell[0]] for cell in G.edges]
    edge_width = np.asarray([G.edges[e]['weight'] for e in G.edges()])
    edge_width = scale_list(edge_width, max_val=edge_width_scale[1], min_val=edge_width_scale[0])

    node_color = [cell_type_colors[cell] for cell in G.nodes]
    node_size = pivot_table.sum(axis=1).values
    node_size = scale_list(node_size, max_val=node_size_scale[1], min_val=node_size_scale[0])


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