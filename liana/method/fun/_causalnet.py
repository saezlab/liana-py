import pandas as pd

def _check_if_corneto():
    try:
        import corneto
    except Exception as e:
        raise ImportError("CORNETO is not correctly installed. Please install it with: "
                          "'pip install git+https://github.com/saezlab/corneto.git@0.9.1-alpha.3 cvxpy==1.3.1 cylp==0.91.5'. "
                          "GUROBI solver is recommended, free academic licenses are available at https://www.gurobi.com/academia/academic-program-and-licenses/.", str(e))
    return corneto


def build_prior_network(ppis, input_nodes, output_nodes, lr_sep=None, verbose=True):
    cn = _check_if_corneto()

    if lr_sep is not None:
        if any(lr_sep in k for k in input_nodes.keys()):
            _print(f"Input nodes are in the format Ligand{lr_sep}Receptor. Extracting only the receptor...", verbose=verbose)
            # Print only at most the first 3 entries
            for k, v in list(input_nodes.items())[:3]:
                _print(f" - {k} -> {v}", verbose=verbose)
            _print(" - ...", verbose=verbose)
            # Do the split only if lr_sep is present, otherwise get the same key:
            input_nodes = {k.split(lr_sep)[-1]: v for k, v in input_nodes.items()}

    _print("Importing network...", verbose=verbose)
    if isinstance(ppis, list):
        G = cn.Graph.from_sif_tuples(ppis)
    elif isinstance(ppis, pd.DataFrame):
        ppis = [(r.source, r.mor, r.target) for (_, r) in ppis.iterrows() if r.source != r.target]
        G = cn.Graph.from_sif_tuples(ppis)
    else:
        raise ValueError("PPIs must be a list of tuples or a pandas DataFrame.")
    _print(f"done.", verbose=verbose)
    
    incl_inputs = set(G.vertices).intersection(set(input_nodes))
    incl_outputs = set(G.vertices).intersection(set(output_nodes))

    _print(f" - Nodes x Edges: {G.num_vertices, G.num_edges}", verbose=verbose)
    _print(f" - Provided inputs included in the prior network: {len(incl_inputs)}/{len(input_nodes)}", verbose=verbose)
    _print(f" - Provided outputs included in the network: {len(incl_outputs)}/{len(output_nodes)}", verbose=verbose)
    
    _print("Performing reachability analysis...", verbose=verbose)
    Gp = G.prune(list(incl_inputs), list(incl_outputs))
    _print("done.", verbose=verbose)
    pruned_size = (Gp.num_vertices, Gp.num_edges)
    incl_inputs_pruned = set(Gp.vertices).intersection(incl_inputs)
    incl_outputs_pruned = set(Gp.vertices).intersection(incl_outputs)
    
    _print(f" - Selected inputs: {len(incl_inputs_pruned)}/{len(incl_inputs)}.", verbose=verbose)
    _print(f" - Selected outputs: {len(incl_outputs_pruned)}/{len(incl_outputs)}.", verbose=verbose)
    _print(f" - Final size of the prior graph: {pruned_size}.", verbose=verbose)
    
    if len(incl_outputs_pruned) == 0:
        raise ValueError("None of the output nodes can be reached from the provided input nodes in the PPI network.")
    if len(incl_inputs_pruned) == 0:
        raise ValueError("None of the input nodes can reach any of the output nodes in the PPI network.")
    
    return Gp


def _print(*args, verbose=True):
    if verbose:
        print(*args)

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
        max_seconds=None,
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
        A dictionary of node weights. The keys are the node names, the values are the weeights.
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
    max_seconds : int, optional
        The maximum number of seconds to run the solver. Default: None
    verbose : bool, optional 
        Whether to print progress information. Default: True
    **kwargs : dict, optional
        Additional arguments to pass to the solver.
    """
    
    
    cn = _check_if_corneto()

    if solver is None:
        solver = cn.methods.carnival.select_mip_solver()

    measured_nodes = set(input_node_scores.keys()) | set(output_node_scores.keys())
 
    _print("Total positive/negative scores of the inputs and outputs:", verbose=verbose)
    w_neg_in, w_pos_in = _get_scores(input_node_scores)
    w_neg_out, w_pos_out = _get_scores(output_node_scores)
    _print(f" - (-) input nodes: {sum(w_neg_in)}", verbose=verbose)
    _print(f" - (+) input nodes: {sum(w_pos_in)}", verbose=verbose)
    _print(f" - (-) output nodes: {sum(w_neg_out)}", verbose=verbose)
    _print(f" - (+) output nodes: {sum(w_pos_out)}", verbose=verbose)
    
    # Total weights
    total = abs(sum(w_neg_in)) + abs(sum(w_neg_out)) + sum(w_pos_in) + sum(w_pos_out)
    _print(f" - abs total (inputs + outputs): {total}", verbose=verbose)
    
    if node_weights is None:
        node_weights = {}
    else:
        node_penalties = _weights_to_penalties(node_weights,
                                               cutoff=node_cutoff,
                                               max_penalty=max_penalty,
                                               min_penalty=min_penalty)

    # assign 0 penalties to input/output nodes, missing_penalty to missing nodes
    c_node_penalties = {k: node_penalties.get(k, missing_penalty) if k not in measured_nodes else 0.0 for k in prior_graph.vertices}

    _print("Building CORNETO problem...", verbose=verbose)
    P, G = cn.methods.carnival._extended_carnival_problem(
        prior_graph,
        input_node_scores,
        output_node_scores,
        node_penalties=c_node_penalties,
        edge_penalty=edge_penalty
    )

    _print(f"Solving with {solver}...", verbose=verbose)
    ps = P.solve(
        solver=solver, 
        max_seconds=max_seconds, 
        verbosity=int(verbose),
        scipy_options=dict(disp='true'), 
        **kwargs)
    
    _print("Done.", verbose=verbose)
    
    obj_names = ["Loss (unfitted inputs/output)", "Edge penalty error", "Node penalty error"]
    _print("Solution summary:", verbose=verbose)
    for s, o in zip(obj_names, P.objectives):
        _print(f" - {s}: {o.value}", verbose=verbose)
    rows, cols = cn.methods.carnival.export_results(P, G, input_node_scores, output_node_scores)
    df = pd.DataFrame(rows, columns=cols)
    return df, P


def _weights_to_penalties(props, 
                          cutoff,
                          min_penalty, 
                          max_penalty):
    if any(p < 0 or p > 1 for p in props.values()):
        raise ValueError("Node weights were not between 0 and 1. Consider minmax or another normalization.")
    
    return {k: max_penalty if v < cutoff else min_penalty for k, v in props.items()}
