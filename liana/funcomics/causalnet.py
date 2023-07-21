import warnings
import pandas as pd
import numpy as np


def _check_if_corneto():
    try:
        import corneto
    except Exception as e:
        raise ImportError("CORNETO is not correctly installed. Please install it with: "
                          "'pip install git+https://github.com/saezlab/corneto.git@0.9.1-alpha.0 cvxpy==1.3.1 cylp==0.91.5'. "
                          "GUROBI solver is recommended, free academic licenses are available at https://www.gurobi.com/academia/academic-program-and-licenses/.", str(e))
    return corneto


def build_prior_network(ppis, input_nodes, output_nodes, lr_sep=None, verbose=True):
    v = verbose
    cn = _check_if_corneto()

    if lr_sep is not None:
        if any(lr_sep in k for k in input_nodes.keys()):
            _print(f"Input nodes are in the format Ligand{lr_sep}Receptor. Extracting only the receptor...", v=v)
            # Print only at most the first 3 entries
            for k, v in list(input_nodes.items())[:3]:
                _print(f" - {k} -> {v}", v=v)
            _print(" - ...", v=v)
            # Do the split only if lr_sep is present, otherwise get the same key:
            input_nodes = {k.split(lr_sep)[-1]: v for k, v in input_nodes.items()}

    _print("Importing network...", end='', v=v)
    if isinstance(ppis, list):
        G = cn.Graph.from_sif_tuples(ppis)
    elif isinstance(ppis, pd.DataFrame):
        ppis = [(r.source, r.mor, r.target) for (_, r) in ppis.iterrows() if r.source != r.target]
        G = cn.Graph.from_sif_tuples(ppis)
    else:
        raise ValueError("PPIs must be a list of tuples or a pandas DataFrame.")
    _print(f"done.", v=v)
    
    incl_inputs = set(G.vertices).intersection(set(input_nodes))
    incl_outputs = set(G.vertices).intersection(set(output_nodes))

    _print(f" - Nodes x Edges: {G.num_vertices, G.num_edges}", v=v)
    _print(f" - Provided inputs included in the prior network: {len(incl_inputs)}/{len(input_nodes)}", v=v)
    _print(f" - Provided outputs included in the network: {len(incl_outputs)}/{len(output_nodes)}", v=v)
    
    _print("Performing reachability analysis...", end='', v=v)
    Gp = G.prune(list(incl_inputs), list(incl_outputs))
    _print("done.", v=v)
    pruned_size = (Gp.num_vertices, Gp.num_edges)
    incl_inputs_pruned = set(Gp.vertices).intersection(incl_inputs)
    incl_outputs_pruned = set(Gp.vertices).intersection(incl_outputs)
    
    _print(f" - Selected inputs: {len(incl_inputs_pruned)}/{len(incl_inputs)}.", v=v)
    _print(f" - Selected outputs: {len(incl_outputs_pruned)}/{len(incl_outputs)}.", v=v)
    _print(f" - Final size of the prior graph: {pruned_size}.", v=v)
    
    if len(incl_outputs_pruned) == 0:
        raise ValueError("None of the output nodes can be reached from the provided input nodes in the PPI network.")
    if len(incl_inputs_pruned) == 0:
        raise ValueError("None of the input nodes can reach any of the output nodes in the PPI network.")
    
    return Gp


def _print(*args, **kwargs):
    # TODO: To be replaced by a logger
    v = kwargs.get('v', None)
    if v is not None:
        kwargs.pop('v')
        print(*args, **kwargs)

def _get_scores(d):
    return (
       [v for v in d.values() if v < 0],
       [v for v in d.values() if v > 0]
    )


def search_causalnet(
        prior_graph,
        input_node_scores,
        output_node_scores,
        node_weights=None,
        node_cutoff=0.1,
        min_penalty=0.01, 
        max_penalty=1.0,
        missing_penalty=10,
        edge_penalty=1e-2,
        solver=None,
        verbose=True,
        show_solver_output=False,
        max_seconds=None,
        **kwargs
    ):
    v = verbose
    cn = _check_if_corneto()

    if solver is None:
        solver = cn.methods.carnival.select_mip_solver()

    # If keys are Ligand^Receptor, create a new dict only with the receptor part:
    # _input = {k.split("^")[1]: v for k, v in input_node_scores.items()}
    measured_nodes = set(input_node_scores.keys()) | set(output_node_scores.keys())
 
    _print("Total positive/negative scores of the inputs and outputs:", v=verbose)
    w_neg_in, w_pos_in = _get_scores(input_node_scores)
    w_neg_out, w_pos_out = _get_scores(output_node_scores)
    _print(f" - (-) input nodes: {sum(w_neg_in)}", v=v)
    _print(f" - (+) input nodes: {sum(w_pos_in)}", v=v)
    _print(f" - (-) output nodes: {sum(w_neg_out)}", v=v)
    _print(f" - (+) output nodes: {sum(w_pos_out)}", v=v)
    
    # Total weights
    total = abs(sum(w_neg_in)) + abs(sum(w_neg_out)) + sum(w_pos_in) + sum(w_pos_out)
    _print(f" - abs total (inputs + outputs): {total}", v=v)
    
    if node_weights is None:
        node_weights = {}
    else:
        node_penalties = _weights_to_penalties(node_weights, cutoff=node_cutoff,
                                               max_penalty=max_penalty,
                                               min_penalty=min_penalty)

    # assign 0 penalties to input/output nodes, missing_penalty to missing nodes
    c_node_penalties = {k: node_penalties.get(k, missing_penalty) if k not in measured_nodes else 0.0 for k in prior_graph.vertices}

    _print("Building CORNETO problem...", v=v)
    P, G = cn.methods.carnival._extended_carnival_problem(
        prior_graph,
        input_node_scores,
        output_node_scores,
        node_penalties=c_node_penalties,
        edge_penalty=edge_penalty
    )

    _print(f"Solving with {solver}...", v=v)
    ps = P.solve(
        solver=solver, 
        max_seconds=max_seconds, 
        verbosity=int(show_solver_output),
        scipy_options=dict(disp='true'), 
        **kwargs)
    
    _print("Done.", v=verbose)
    
    obj_names = ["Loss (unfitted inputs/output)", "Edge penalty error", "Node penalty error"]
    _print("Solution summary:", v=v)
    for s, o in zip(obj_names, P.objectives):
        _print(f" - {s}: {o.value}", v=v)
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