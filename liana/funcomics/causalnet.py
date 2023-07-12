import warnings
import pandas as pd
import numpy as np



def _check_if_corneto():
    try:
        import corneto
    except Exception as e:
        raise ImportError("CORNETO is not correctly installed. Please install it with: "
                          "'pip install git+https://github.com/saezlab/corneto.git@0.9.1-alpha.0 cvxpy==1.3.1 cylp==0.91.5 gurobipy'. "
                          "GUROBI solver is recommended, free academic licenses are available at https://www.gurobi.com/academia/academic-program-and-licenses/.", str(e))
    return corneto


def _check_graphviz():
    try:
        import graphviz
    except Exception as e:
        return ImportError("Graphviz not installed, but required for plotting. Please install it using conda: 'conda install python-graphviz.'", str(e))   
    return graphviz


def _run_carnival(perturbations, measurements, priorKnowledgeNetwork,
                  node_penalty, default_edge_penalty=1e-4,
                  maximize_inclusion_receptors=False,
                  solver=None, verbosity=False, **kwargs):
    corneto = _check_if_corneto()
    data = dict()
    for k, v in perturbations.items():
        data[k] = ("P", v)
    for k, v in measurements.items():
        data[k] = ("M", v)
    conditions = {"c0": data}
    if isinstance(priorKnowledgeNetwork, list):
        G = corneto.Graph.from_sif_tuples(priorKnowledgeNetwork)
    elif isinstance(priorKnowledgeNetwork, corneto.Graph):
        G = priorKnowledgeNetwork
    else:
        raise ValueError("Provide a list of sif tuples or a graph")
    Gf = corneto.methods.create_flow_graph(G, conditions)
    P = corneto.methods.signflow.signflow(Gf, conditions, l0_penalty_edges=default_edge_penalty,
                                          flow_implies_signal=maximize_inclusion_receptors)
    
    selected_nodes = P.symbols['species_activated_c0'] + P.symbols['species_activated_c0']
    node_weights = np.array([node_penalty[n] if not n.startswith('_') else 0.0 for n in Gf.vertices])
    # If not selected and have weight, increase the error by the weight of the node
    P.add_objectives(node_weights @ selected_nodes, inplace=True)
    P.solve(solver=solver, verbosity=int(verbosity), **kwargs)
    return P, Gf


def _get_node_props(prefix, row):
    props = dict(shape='circle')
    name = prefix
    pred_val = prefix + '_pred_val'
    node_type = prefix + '_type'

    if len(row[name]) == 0:
        props['shape'] = 'point'
    if row[pred_val] != 0:
        props['penwidth'] = '2'
        props['style'] = 'filled'
    if row[pred_val] > 0:
        props['color'] = 'red'
        props['fillcolor'] = 'lightcoral'
    if row[pred_val] < 0:
        props['color'] = 'blue'
        props['fillcolor'] = 'azure2'
    if row[node_type] == 'input':
        props['shape'] = 'invtriangle'
    if row[node_type] == 'output':
        props['shape'] = 'square'
    return props


def plot_df(df, clean=True, node_attr=None):
    graphviz = _check_graphviz()
    if node_attr is None:
        node_attr = dict(fixedsize="true")
    g = graphviz.Digraph(node_attr=node_attr)
    for i, row in df.iterrows():
        # Create node source and target. If clean, ignore dummy nodes
        if clean:
            if len(row.source) == 0 or row.source.startswith('_'):
                continue
            if len(row.target) == 0 or row.target.startswith('_'):
                continue
            if row.source_pred_val == 0 or row.target_pred_val == 0:
                continue
        edge_props = dict(arrowhead='normal')
        edge_interaction = int(row.edge_type)
        if edge_interaction != 0:
            edge_props['penwidth'] = '2'
        if edge_interaction < 0:
            edge_props['arrowhead'] = 'tee'
        if int(row.edge_pred_val) < 0:
            edge_props['color'] = 'blue'
        if int(row.edge_pred_val) > 0:
            edge_props['color'] = 'red'
        g.node(row.source, **_get_node_props('source', row))
        g.node(row.target, **_get_node_props('target', row))
        g.edge(row.source, row.target, **edge_props)
    return g


def plot(graph, format=None, clean=True):
    graphviz = _check_graphviz()

    if isinstance(format, pd.DataFrame):
        # Convert to a dict
        raise NotImplementedError()

    vertices, edges = graph.vertices, graph.edges
    custom_vertex = dict()
    custom_edge = dict()
    if format:
        # Add custom values per edge/vertex
        for v in vertices:
            value = format.get(v, 0)
            if value < 0:
                custom_vertex[v] = dict(
                    color="blue",
                    penwidth="2",
                    fillcolor="azure2",
                    style="filled"
                )
            elif value > 0:
                custom_vertex[v] = dict(
                    color="red",
                    penwidth="2",
                    fillcolor="lightcoral",
                    style="filled",
                )
            #if clean and value != 0:
            #    custom_vertex[v]['shape'] = 'triangle'
            #else:
            custom_vertex[v]['shape'] = 'circle'
        for e in edges:
            value = format.get(e, 0)
            if value < 0:
                custom_edge[e] = dict(color="blue", penwidth="2")
            elif value > 0:
                custom_edge[e] = dict(color="red", penwidth="2")

    node_attr = dict(fixedsize="true")
    g = graphviz.Digraph(node_attr=node_attr)
    for e, p in zip(edges, graph.edge_properties):
        s, t = e
        s = list(s)
        t = list(t)
        if clean and (len(s) == 0 or s[0].startswith('_') or len(t) == 0 or t[0].startswith('_')):
            continue
        # Check also edges without value
        if clean and format is not None and custom_edge.get(e, dict()).get('color', None) is None:
            continue
        # Also ignore and edge A - B if B is not selected
        if clean and format is not None and custom_vertex.get(t[0], dict()).get('color', None) is None:
            continue
        if len(s) == 0:
            s = f"*_{str(t)}"
            g.node(s, shape="point")
        elif len(s) == 1:
            s = str(s[0])
            props = custom_vertex.get(s, dict())
            g.node(s, **props)
        else:
            raise NotImplementedError("Represent- hyperedges as composite edges")
        if len(t) == 0:
            t = f"{str(s)}_*"
            g.node(t, shape="point")
        elif len(t) == 1:
            t = str(t[0])
            props = custom_vertex.get(t, dict())
            g.node(t, **props)
        edge_type = p.get("interaction", 0)
        props = custom_edge.get(e, dict())
        if edge_type >= 0:
            g.edge(s, t, arrowhead="normal", **props)
        else:
            g.edge(s, t, arrowhead="tee", **props)
    return g


def get_problem_values(graph, problem, condition=None):
    values = dict()
    if condition is None:
        condition = "c0"
    vertex_values = (
        problem.symbols[f"species_activated_{condition}"].value
        - problem.symbols[f"species_inhibited_{condition}"].value
    )
    edge_values = (
        problem.symbols[f"reaction_sends_activation_{condition}"].value
        - problem.symbols[f"reaction_sends_inhibition_{condition}"].value
    )
    for v, val in zip(graph.vertices, vertex_values):
        values[v] = val
    for e, val in zip(graph.edges, edge_values):
        values[e] = val
    return values


def _get_receptors(d, top=None):
    d = dict(sorted(d.items(), key=lambda item: abs(item[1]), reverse=True))
    if top is None:
        top = len(d)
    return {k.split('^')[1]: v for i, (k, v) in enumerate(d.items()) if i < top}


def _get_tfs(d, top=25):
    if top is None:
        top = len(d)
    d = dict(sorted(d.items(), key=lambda item: abs(item[1]), reverse=True))
    return {k: v for i, (k, v) in enumerate(d.items()) if i < top}


def preprocess_ppi(ppis, lr_data, tf_data, gene_weights, int_target, score_lfc_pval=True):
    ppis_pruned = ppis[['source_genesymbol', 'target_genesymbol', 'consensus_stimulation', 'consensus_inhibition']].copy()
    ppis_pruned['mor'] = np.where((ppis_pruned['consensus_stimulation'] == 1) & (ppis_pruned['consensus_inhibition'] == 0), 1, np.where((ppis_pruned['consensus_stimulation'] == 0) & (ppis_pruned['consensus_inhibition'] == 1), -1, 0))
    ppis_pruned = ppis_pruned[ppis_pruned['mor'] != 0][['source_genesymbol', 'mor', 'target_genesymbol']].drop_duplicates()
    ppis_pruned.columns = ['source', 'mor', 'target']
    ### PREPROCESSING ###
    all_network_nodes = np.union1d(ppis_pruned['source'].unique(), ppis_pruned['target'].unique())
    # ligand-receptor pairs
    filt_lr_data = lr_data[(lr_data['sign'] != 'neither') & (lr_data['receptor_complex'].isin(all_network_nodes))].sort_values('interaction_padj').copy()
    filt_lr_data['sign_numeric'] = np.where(filt_lr_data['sign'] == 'positive', 1, -1)
    # create a metric which is the -log10 adjusted p-value multiplied by the sign
    if score_lfc_pval:
        filt_lr_data['lr_metric'] = -np.log10(filt_lr_data['interaction_padj']) * filt_lr_data['sign_numeric']
    else:
        filt_lr_data['lr_metric'] = filt_lr_data['sign_numeric']
    lr_dict = filt_lr_data.set_index('interaction')['lr_metric'].to_dict()
    # gene weights based on gene expression proportion
    int_gene_weights = gene_weights[gene_weights['cell_type'] == int_target].copy()
    int_gene_weights_dict = int_gene_weights.set_index('gene')['prop'].to_dict()
    int_gene_weights_dict = {k: v for k, v in int_gene_weights_dict.items() if k in all_network_nodes}
    # transcription factors
    tfs_in_op = tf_data.columns[tf_data.columns.isin(all_network_nodes)]
    int_tf_vec = tf_data.loc[int_target, tfs_in_op].copy()
    int_tf_dict = int_tf_vec.reindex(int_tf_vec.abs().sort_values(ascending=False).index).to_dict()
    print(f'{len(tf_data.columns) - len(tfs_in_op)} out of {len(tf_data.columns)} tfs are not in the network')
    # from the network, remove all edges on which the 'source' is a tf_in_op
    pk_net_no_tfs = ppis_pruned[~ppis_pruned['source'].isin(tfs_in_op)].copy()
    return {
        'lr_dict': lr_dict,
        'gene_weights_dict': int_gene_weights_dict,
        'tf_dict': int_tf_dict,
        'pk_net': pk_net_no_tfs
    }
    

def preprocess(data, top_receptors=None, top_tfs=10,
               penalty_gene_proportions=None):
    cn = _check_if_corneto()
    if penalty_gene_proportions is None:
        penalty_gene_proportions = dict()
    output_data = _get_tfs(data['tf_dict'], top=top_tfs)
    input_data = _get_receptors(data['lr_dict'], top=top_receptors)
    print(f"Selected top {len(input_data)} receptors")
    print(f"Selected top {len(output_data)} TFs")
    # Check proportions
    counts = sum(1 for value in data['gene_weights_dict'].values() if value < 0 or value > 1)
    if counts > 0:
        raise ValueError(f"Expected gene proportions in [0, 1] for 'gene_weights_dict', but {counts} genes have a value outside this range. Please check if these values are proportions")
    # Import the network
    print("Importing network...", end='')
    sif_tuples_pkn = [(r.source, r.mor, r.target) for (_, r) in data['pk_net'].iterrows() if r.source != r.target]
    nodes_in_pkn = set(s for (s, _, _) in sif_tuples_pkn) | set(t for (_, _, t) in sif_tuples_pkn)
    print(f"done ({len(sif_tuples_pkn)} tuples, {len(nodes_in_pkn)} species)")
    print("Building network...", end='')
    
    G = cn.Graph.from_sif_tuples(sif_tuples_pkn)
    orig_size = (G.num_vertices, G.num_edges)
    print(f"done (VxE: {orig_size})")
    # Check how many perturbations and measurements are in the PKN
    incl_perts = set(G.vertices).intersection(set(input_data.keys()))
    incl_meas = set(G.vertices).intersection(set(output_data.keys()))
    print(f"Number of inputs/output mapped to the network: {len(incl_perts)}/{len(input_data)} inputs (receptors), {len(incl_meas)}/{len(output_data)} outputs (TFs).")
    # Check how many perturbations/measurements are after prunning
    print("Performing reachability analysis...", end='')
    Gp = G.prune(list(input_data.keys()), list(output_data.keys()))
    print("done.")
    pruned_size = (Gp.num_vertices, Gp.num_edges)
    incl_perts_p = set(Gp.vertices).intersection(set(input_data.keys()))
    incl_meas_p = set(Gp.vertices).intersection(set(output_data.keys()))
    input_data_pruned = {k: input_data[k] for k in incl_perts_p}
    output_data_pruned = {k: output_data[k] for k in incl_meas_p}
    print(f" - Selected inputs: {len(incl_perts_p)}/{len(input_data)} (receptors).")
    print(f" - Selected outputs: {len(incl_meas_p)}/{len(output_data)} (TFs).")
    print(f" - Graph final size (VxE): {pruned_size}")
    if len(incl_meas_p) == 0:
        raise ValueError("None of the output nodes can be reached from the input nodes in the PKN")
    if len(incl_perts_p) == 0:
        raise ValueError("None of the input nodes can reach any of the output nodes in the PKN")
    
    # Process data
    p = penalty_gene_proportions.get('prop_threshold', 0.75)
    p_below = penalty_gene_proportions.get('penalty_below', 1e-1)
    p_above = penalty_gene_proportions.get('penalty_above', 1e-3)
    p_unkno = penalty_gene_proportions.get('penalty_unknown', 5.0)
    
    n_weights_d = dict()
    t_m, t_a, t_b = 0, 0, 0
    p_m, p_a, p_b = 0, 0, 0

    for v in Gp.vertices:
        w = p_unkno
        if v in incl_meas_p or v in incl_perts_p:
            n_weights_d[v] = 0
            continue
        prop_gene = data['gene_weights_dict'].get(v, None)
        if prop_gene is None:
            t_m += 1
            p_m += w
        elif prop_gene <= p:
            w = p_below
            t_b += 1
            p_b += w
        else:
            w = p_above
            t_a += 1
            p_a += w
        n_weights_d[v] = w

    print(f"Number of unknown/above/below node proportions (threshold {p}):", (t_m, t_a, t_b))
    print("Total assigned penalties to unknown/above/below nodes:", (p_m, p_a, p_b))
    print("Total absolute input scores (receptors):", sum(abs(v) for v in input_data_pruned.values()))
    print("Total absolute output scores (tfs):", sum(abs(v) for v in output_data_pruned.values()))
    return G, Gp, input_data_pruned, output_data_pruned, n_weights_d


def export_df(P, Gf, ip_d, out_d):
    # Get results
    nodes = P.symbols['species_activated_c0'].value - P.symbols['species_inhibited_c0'].value
    edges = P.symbols['reaction_sends_activation_c0'].value - P.symbols['reaction_sends_inhibition_c0'].value
    
    # For all edges
    E = Gf.edges
    V = {v: i for i, v in enumerate(Gf.vertices)}

    df_rows = []
    for i, e in enumerate(E):
        s, t = e
        if len(s) > 0:
            s = list(s)[0]
        else:
            s = ''
        if len(t) > 0:
            t = list(t)[0]
        else:
            t = ''
        if abs(edges[i]) > 0.5:
            # Get value of source/target
            edge_type = Gf.edge_properties[i].get('interaction', 0)
            s_val = nodes[V[s]] if s in V else 0
            t_val = nodes[V[t]] if t in V else 0
            s_type = 'unmeasured'
            s_weight = 0
            t_type = 'unmeasured'
            t_weight = 0
            if s in ip_d:
                s_type = 'input'
                s_weight = ip_d.get(s)
            if t in out_d:
                t_type = 'output'
                t_weight = out_d.get(t)

            df_rows.append([s, s_type, s_weight, s_val, t, t_type, t_weight, t_val, edge_type, edges[i]])
    df = pd.DataFrame(df_rows, columns=['source', 'source_type', 'source_weight', 'source_pred_val',
                                        'target', 'target_type', 'target_weight', 'target_pred_val',
                                        'edge_type', 'edge_pred_val'])
    return df


def _select_solver():
    cn = _check_if_corneto()
    priority = ['gurobi', 'cplex', 'copt', 'mosek', 'scipy', 'scip', 'cbc', 'glpk_mi', None]
    solvers = [s.lower() for s in cn.K.available_solvers()]
    solver = None
    for i, solver in enumerate(priority):
        if solver is None:
            raise ValueError(f"No valid MIP solver installed, solvers detected: {solvers}")
        if solver in solvers:
            solver = solver.upper()
            if i > 3:
                warnings.warn(f"Note: {solver} has been selected as the default solver. However, for optimal performance, it's recommended to use one of the following solvers: GUROBI, CPLEX, COPT, or MOSEK.")
            break
    return solver


def find_network(
        data, top_receptors=None, top_tfs=50,
        penalty_gene_proportions={'prop_threshold': 0.75, 'penalty_below': 1e-1,
                                  'penalty_above': 1e-3, 'penalty_unknown': 5.0},
        penalty_edges=1e-3,
        maximize_inclusion_receptors=False,
        solver=None,
        verbose=True,
        **kwargs
):
    
    G, Gp, ip_d, out_d, n_w, = preprocess(
        data, top_receptors=top_receptors,
        top_tfs=top_tfs,
        penalty_gene_proportions=penalty_gene_proportions
    )
    print(f"Penalty on edges: {penalty_edges} (for {Gp.num_edges} edges), total={Gp.num_edges * penalty_edges}")
    
    if solver is None:
        solver = _select_solver()

    print(f"Solving problem with {solver}...")
    P, Gf = _run_carnival(
        ip_d,
        out_d,
        Gp,
        node_penalty=n_w,
        default_edge_penalty=penalty_edges,
        maximize_inclusion_receptors=maximize_inclusion_receptors,
        solver=solver,
        verbosity=verbose,
        **kwargs
    )
    descr = ["Total output error (unfitted TFs)",
             "Node penalty (non-removed penalized nodes)",
             "Edge penalty (non-removed edges)"]
    
    print("Solution found:")
    for o, s in zip(P.objectives, descr):
        print(f" - Objective {s}:", o.value)

    df = export_df(P, Gf, ip_d, out_d)
    return df, P, Gf


def carnival_network(input_nodes, output_nodes, pkn, node_penalty=1e-3, 
                     maximize_inclusion_receptors=True, solver=None, 
                     verbose=True, **kwargs):
    """
    Generate a sign consistent subgraph from a prior knowledge network (PKN) from input nodes (e.g receptors) to output nodes
    (e.g transcription factors).
    
    It solves a constrained optimization problem to find a parsimonious network 
    consistent both with prior knowledge and data.

    
    Parameters
    ----------
    input_nodes : dict
        Dictionary with input nodes. These nodes have to be present in the PKN, 
        otherwise ignored. Values correspond to the weight (importance) of the node.
        Negative values indicate downregulation or inactivation, and positive values indicate
        upregulation or activation.
    output_nodes : pandas.DataFrame
        Dictionary with output nodes downstream the input nodes. Weights indicate the importance
        and activity of the node, as in the case of input_nodes.
    pkn : list
        List with tuples of interactions in the prior knowledge, where ('A', 1, 'B') indicates an
        activatory activation from A to B, and ('A', -1, 'B') an inhibitory interaction.
    node_penalty : int, list or dict, optional
    solver : str, optional
        Name of the solver: GUROBI, CPLEX, MOSEK, CBC, SCIP, GLPK_MI
    verbose : bool, optional
        If true, prints information from the solver during the optimization process.
    
    Returns
    -------
    A tuple (dataframe, problem, graph), where:
    - dataframe contains the selected networks, with columns:
        - source: source node of the edge 
        - edge_sign: 1 if activatory edge, -1 if inhibitory
        - target: target node of the edge
        - value_source: predicted value (0: not used, 1: activation, -1: inhibition)
        - value_edge: predicted value for the edge (independent of its sign) (0: none, 1: activates target node, -1: inhibits target node) 
        - value_target: predicted value for the target node
    - problem: the solved CORNETO optimization problem
    - graph: extended graph from the pkn with extra dummy nodes required in CORNETO
    """
    cn = _check_if_corneto()
    verbosity = int(verbose)
    # If perturbations is not a dict, transform to a dict
    if not isinstance(input_nodes, dict):
        warnings.warn("Input nodes do not have a known sign, assuming that they can be -1 or +1 (unknown).")
        input_nodes = {v: 0 for v in input_nodes}
    if not isinstance(output_nodes, dict):
        raise ValueError(f"Expected type for output nodes is dict but got {type(output_nodes)} instead.")
    if solver is None:
        priority = ['gurobi', 'cplex', 'copt', 'mosek', 'scipy', 'scip', 'cbc', 'glpk_mi', None]
        solvers = [s.lower() for s in cn.K.available_solvers()]
        for i, solver in enumerate(priority):
            if solver is None:
                raise ValueError(f"No valid MIP solver installed, solvers detected: {solvers}")
            if solver in solvers:
                solver = solver.upper()
                if i > 4:
                   warnings.warn(f"Note: {solver} has been selected as the default solver. However, for optimal performance, it's recommended to use one of the following solvers: GUROBI, CPLEX, COPT, or MOSEK.")
                break
    if isinstance(pkn, pd.DataFrame):
        warnings.warn("Assuming the dataframe has three columns, where col0=source, col1=interaction, col2=target.")
        pkn = [(r[0], r[1], r[2]) for (i, r) in pkn.iterrows()]
    elif isinstance(pkn, list):
        if len(pkn[0]) != 3:
            raise ValueError("The Prior Knowledge Network has to be a list of tuples (source, interaction, target)"
                             ", e.g [('A', 1, 'B'), ('A', -1, 'C'), ...] or a DataFrame with three columns.")
    G = cn.Graph.from_sif_tuples(pkn)
    orig_size = (G.num_vertices, G.num_edges)
    print(f"Graph shape before pruning: {orig_size}")
    # Check how many perturbations and measurements are in the PKN
    incl_perts = set(G.vertices).intersection(set(input_nodes.keys()))
    incl_meas = set(G.vertices).intersection(set(output_nodes.keys()))
    print(f"Inputs: {len(incl_perts)}, outputs: {len(incl_meas)}")
    # Check how many perturbations/measurements are after prunning
    Gp = G.prune(list(input_nodes), list(output_nodes.keys()))
    pruned_size = (Gp.num_vertices, Gp.num_edges)
    incl_perts_p = set(Gp.vertices).intersection(set(input_nodes.keys()))
    incl_meas_p = set(Gp.vertices).intersection(set(output_nodes.keys()))
    print(f"Inputs after pruning: {len(incl_perts_p)}, outputs after pruning: {len(incl_meas_p)}")
    print(f"Graph shape after pruning: {pruned_size}")
    if len(incl_meas_p) == 0:
        raise ValueError("None of the output nodes can be reached from the input nodes in the PKN")
    if len(incl_perts_p) == 0:
        raise ValueError("None of the input nodes can reach any of the output nodes in the PKN")
        
    

    P, Gf = _run_carnival(cn, input_nodes, output_nodes, Gp, node_penalty=node_penalty, 
                          
                          maximize_inclusion_receptors=maximize_inclusion_receptors, solver=solver,
                          verbosity=verbosity, **kwargs)
    # Get results
    nodes = P.symbols['species_activated_c0'].value - P.symbols['species_inhibited_c0'].value
    edges = P.symbols['reaction_sends_activation_c0'].value - P.symbols['reaction_sends_inhibition_c0'].value
    
    # For all edges
    E = Gf.edges
    V = {v: i for i, v in enumerate(Gf.vertices)}

    df_rows = []
    for i, e in enumerate(E):
        s, t = e
        if len(s) > 0:
            s = list(s)[0]
        else:
            s = ''
        if len(t) > 0:
            t = list(t)[0]
        else:
            t = ''
        if abs(edges[i]) > 0.5:
            # Get value of source/target
            edge_type = Gf.edge_properties[i].get('interaction', 0)
            s_val = nodes[V[s]] if s in V else 0
            t_val = nodes[V[t]] if t in V else 0
            s_type = 'unmeasured'
            s_weight = 0
            t_type = 'unmeasured'
            t_weight = 0
            if s in input_nodes:
                s_type = 'input'
                s_weight = input_nodes.get(s)
            if t in output_nodes:
                t_type = 'output'
                t_weight = output_nodes.get(t)

            df_rows.append([s, s_type, s_weight, s_val, t, t_type, t_weight, t_val, edge_type, edges[i]])
    df = pd.DataFrame(df_rows, columns=['source', 'source_type', 'source_weight', 'source_pred_val',
                                        'target', 'target_type', 'target_weight', 'target_pred_val',
                                        'edge_type', 'edge_pred_val'])
    return df, P, Gf