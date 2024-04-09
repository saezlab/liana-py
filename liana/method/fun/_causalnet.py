import pandas as pd
import numpy as np
from liana._logging import _logg, _check_if_installed
from liana._constants import DefaultValues as V


def build_prior_network(ppis,
                        input_nodes,
                        output_nodes,
                        lr_sep=None,
                        verbose=V.verbose):
    """
    Build Prior Network from PPIs and input/output nodes.

    Parameters
    ----------
    ppis : list of tuples or pandas DataFrame
        The PPIs to use for the prior network. If a pandas DataFrame is provided, it must have the columns
    input_nodes : dict
        A dictionary of input nodes. The keys are the node names, the values are the node scores.
    output_nodes : dict
        A dictionary of output nodes. The keys are the node names, the values are the node scores.
    lr_sep : str, optional
        The separator to use to split the input nodes into ligand and receptor. If None, the input nodes will be used as is.
    verbose : bool, optional
        Whether to print progress information. Default: True

    Returns
    -------
    corneto.Graph

    """

    cn = _check_if_installed("corneto")

    if lr_sep is not None:
        if any(lr_sep in k for k in input_nodes.keys()):
            _logg(f"Input nodes are in the format Ligand{lr_sep}Receptor. Extracting only the receptor...", verbose=verbose)
            # Print only at most the first 3 entries
            for k, v in list(input_nodes.items())[:3]:
                _logg(f" - {k} -> {v}", verbose=verbose)
            _logg(" - ...", verbose=verbose)
            # Do the split only if lr_sep is present, otherwise get the same key:
            input_nodes = {k.split(lr_sep)[-1]: v for k, v in input_nodes.items()}

    _logg("Importing network...", verbose=verbose)
    if isinstance(ppis, list):
        G = cn.Graph.from_sif_tuples(ppis)
    elif isinstance(ppis, pd.DataFrame):
        ppis = [(r.source, r.mor, r.target) for (_, r) in ppis.iterrows() if r.source != r.target]
        G = cn.Graph.from_sif_tuples(ppis)
    else:
        raise ValueError("PPIs must be a list of tuples or a pandas DataFrame.")
    _logg("done.", verbose=verbose)

    incl_inputs = set(G.vertices).intersection(set(input_nodes))
    incl_outputs = set(G.vertices).intersection(set(output_nodes))

    _logg(f" - Nodes x Edges: {G.num_vertices, G.num_edges}", verbose=verbose)
    _logg(f" - Provided inputs included in the prior network: {len(incl_inputs)}/{len(input_nodes)}", verbose=verbose)
    _logg(f" - Provided outputs included in the network: {len(incl_outputs)}/{len(output_nodes)}", verbose=verbose)

    _logg("Performing reachability analysis...", verbose=verbose)
    Gp = G.prune(list(incl_inputs), list(incl_outputs))
    _logg("done.", verbose=verbose)
    pruned_size = (Gp.num_vertices, Gp.num_edges)
    incl_inputs_pruned = set(Gp.vertices).intersection(incl_inputs)
    incl_outputs_pruned = set(Gp.vertices).intersection(incl_outputs)

    _logg(f" - Selected inputs: {len(incl_inputs_pruned)}/{len(incl_inputs)}.", verbose=verbose)
    _logg(f" - Selected outputs: {len(incl_outputs_pruned)}/{len(incl_outputs)}.", verbose=verbose)
    _logg(f" - Final size of the prior graph: {pruned_size}.", verbose=verbose)

    if len(incl_outputs_pruned) == 0:
        raise ValueError("None of the output nodes can be reached from the provided input nodes in the PPI network.")
    if len(incl_inputs_pruned) == 0:
        raise ValueError("None of the input nodes can reach any of the output nodes in the PPI network.")

    return Gp

def _get_scores(d):
    return (
       [v for v in d.values() if v < 0],
       [v for v in d.values() if v > 0]
    )


def find_causalnet(
        prior_graph,
        input_node_scores,
        output_node_scores,
        node_weights=None,
        node_cutoff=0.1,
        min_penalty=0.01,
        max_penalty=1.0,
        missing_penalty=10,
        edge_penalty=0.01,
        solver=None,
        seed=1337,
        max_runs=1,
        stable_runs=5,
        verbose=True,
        **kwargs
        ):
    """
    Find the causal network that best explains the input/output node scores.

    Parameters
    ----------
    prior_graph : corneto.Graph
        The prior graph to use for the search.
    input_node_scores : dict
        A dictionary of input node scores.
    output_node_scores : dict
        A dictionary of output node scores.
    node_weights : dict, optional
        A dictionary of node weights. The keys are the node names, the values are the weights.
        If None, all nodes will have the same weight.
    node_cutoff : float
        The cutoff to use for the node weights. Nodes with a weight below this cutoff will be assigned
        the max_penalty, nodes with a weight above this cutoff will be assigned the min_penalty.
        Only used if node_weights is not None. Default: 0.1
    min_penalty : float
        The minimum penalty to assign to nodes with a weight above the cutoff.
        Only used if node_weights is not None. Default: 0.01
    max_penalty : float
        The maximum penalty to assign to nodes with a weight below the cutoff
        Only used if node_weights is not None. Default: 1.0
    missing_penalty : float
        The penalty to assign to nodes that are not measured. Default: 10
    edge_penalty : float
        The penalty to assign to edges. Default: 0.01
    solver : str, optional
        The solver to use. If None, the default solver will be used. Default: None
        It will default to the solver included in SCIPY, if no other solver is available.
    seed : int, optional
        The seed to use for the random number generator. Default: 1337
    max_runs : int, optional
        The maximum number of runs to perform. Consider increasing this value if the solver does not converge.
        In each run, the noise added to the edge and node penalties is perturbed slightly (iterating over the seed).
        By default, only 1 run is performed.
    stable_runs : int, optional
        The number of consecutive stable solutions requires to interrupt the iteration over max_runs. Only used if max_runs is not == 1. Default: 5
    verbose : bool, optional
        Whether to print progress information. Default: True
    **kwargs : dict, optional
        Additional arguments to pass to the solver.
    """

    cn = _check_if_installed("corneto")

    if solver is None:
        solver = cn.methods.carnival.select_mip_solver()

    measured_nodes = set(input_node_scores.keys()) | set(output_node_scores.keys())

    _logg("Total positive/negative scores of the inputs and outputs:", verbose=verbose)
    w_neg_in, w_pos_in = _get_scores(input_node_scores)
    w_neg_out, w_pos_out = _get_scores(output_node_scores)
    _logg(f" - (-) input nodes: {sum(w_neg_in)}", verbose=verbose)
    _logg(f" - (+) input nodes: {sum(w_pos_in)}", verbose=verbose)
    _logg(f" - (-) output nodes: {sum(w_neg_out)}", verbose=verbose)
    _logg(f" - (+) output nodes: {sum(w_pos_out)}", verbose=verbose)

    # Total weights
    total = abs(sum(w_neg_in)) + abs(sum(w_neg_out)) + sum(w_pos_in) + sum(w_pos_out)
    _logg(f" - abs total (inputs + outputs): {total}", verbose=verbose)

    if node_weights is None:
        node_penalties = {}
    else:
        node_penalties = _weights_to_penalties(node_weights,
                                               cutoff=node_cutoff,
                                               max_penalty=max_penalty,
                                               min_penalty=min_penalty)

    run_count = 0 # total runs
    stable_count = 0 # stable solutions in a row
    df_all = None # df with all solutions

    while run_count < max_runs:
        current_seed = seed + run_count
        if run_count > 0:
            _logg(f"Run {run_count} with seed {current_seed}", verbose=verbose)

        # assign 0 penalties to input/output nodes, missing_penalty to missing nodes
        # add a small amount of noise to the penalties to ensure reproducible solutions
        rng = np.random.default_rng(seed=current_seed)
        c_node_penalties = {k: node_penalties.get(k, missing_penalty) + rng.uniform(min_penalty/20, min_penalty/10)
                            if k not in measured_nodes else 0.0 for k in prior_graph.vertices}

        _logg("Building CORNETO problem...", verbose=verbose)
        P, G = cn.methods.carnival._extended_carnival_problem(
            prior_graph,
            input_node_scores,
            output_node_scores,
            node_penalties=c_node_penalties,
            edge_penalty=edge_penalty
        )

        # E is the variable with 1 if edge activates or inhibits, 0 otherwise
        E = P.symbols['reaction_sends_activation_c0'] + P.symbols['reaction_sends_inhibition_c0']
        W = rng.uniform(edge_penalty / 20, edge_penalty / 10, size=E.shape)
        P.add_objectives(W.T @ E)

        _logg(f"Solving with {solver}...", verbose=verbose)
        if (solver=='scipy') and verbose:
            kwargs.update(scipy_options=dict(disp='true'))

        P.solve(
            solver=solver,
            verbosity=int(verbose),
            **kwargs)

        obj_names = ["Loss (unfitted inputs/output)", "Edge penalty error", "Node penalty error"]
        _logg("Solution summary:", verbose=verbose)
        for s, o in zip(obj_names, P.objectives):
            _logg(f" - {s}: {o.value}", verbose=verbose)

        rows, cols = cn.methods.carnival.export_results(P, G, input_node_scores, output_node_scores)
        df = pd.DataFrame(rows, columns=cols)

        # Check if all rows from df are contained in df_all
        if df_all is None:
            df_all = df
            continue
        else:
            set_df = set(tuple(row) for row in df.values)
            set_df_all = set(tuple(row) for row in df_all.values)

            if set_df.issubset(set_df_all):
                stable_count += 1
            else:
                stable_count = 0

            df_all = pd.concat([df_all, df]).drop_duplicates()

        if stable_count >= stable_runs:
            break

        run_count += 1

    return df_all, P


def _weights_to_penalties(props,
                          cutoff,
                          min_penalty,
                          max_penalty):
    if any(p < 0 or p > 1 for p in props.values()):
        raise ValueError("Node weights were not between 0 and 1. Consider minmax or another normalization.")

    return {k: max_penalty if v < cutoff else min_penalty for k, v in props.items()}
