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


def _run_carnival(corneto, perturbations, measurements, priorKnowledgeNetwork, 
                  node_list=None, betaWeight=0.2, solver=None, **kwargs):
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
    P = corneto.methods.signflow.signflow(Gf, conditions, l0_penalty_vertices=betaWeight)
    if node_list is not None:
        if not isinstance(node_list, dict):
            node_list = {k: 1 for k in node_list}
        int_input = set(perturbations.keys()).intersection(set(node_list.keys()))
        if len(int_input) > 0:
            raise ValueError(f"There is an overlap between nodes in node_list and the list of input nodes: {int_input}")
        int_output = set(measurements.keys()).intersection(set(node_list.keys()))
        if len(int_output) > 0:
            raise ValueError(f"There is an overlap between nodes in node_list and the list of output nodes: {int_output}")
        # TODO: Check if node_list already in perts or measurements
        selected = P.symbols['species_activated_c0'] + P.symbols['species_activated_c0']
        nodelist_weigths = np.array([node_list.get(n, 0) for n in Gf.vertices])
        # If not selected and have weight, increase the error by the weight of the node
        P.add_objectives(nodelist_weigths @ (1 - selected), inplace=True)
    P.solve(solver=solver, **kwargs)
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
        props['shape'] = 'promoter'
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


def carnival_network(input_nodes, output_nodes, pkn, node_list=None, edge_penalty=0.2, solver=None, verbose=True, **kwargs):
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
    node_list : list or dict, optional
        List or dict with nodes present in the network. If present in the PKN,
        they are prioritized to be included in the network, depending on the weight.
        If a list is provided instead of a dict, each node has a weight of 1, 
        otherwise the weight is taken from the value.
    edge_penalty : float, optional
        Weight of the penalty on the number of selected edges in the network. Increase weight
        to decrease the size of the final network, at expenses of removing also input nodes,
        output nodes, and nodes in node_list. The trade-off is based on the weights of the nodes
        removed from the network.
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
        solvers = [s.lower() for s in cn.K.available_solvers()]
        if 'gurobi' in solvers:
            solver = 'GUROBI'
        elif 'cplex' in solvers:
            solver = 'CPLEX'
        elif 'mosek' in solvers:
            solver = 'MOSEK'
        elif 'scip' in solvers:
            solver = 'SCIP'
        elif 'cbc' in solvers:
            warnings.warn("GUROBI/CPLEX/SCIP not detected, using CBC instead (not recommended)")
            solver = 'CBC'
        elif 'glpk_mi' in solvers:
            warnings.warn("GUROBI/CPLEX/SCIP not detected, using GLPK instead (not recommended)")
            solver = 'GLPK_MI'
        else:
            raise ValueError(f"No valid MIP solver installed, solvers detected: {solvers}")
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
        
    P, Gf = _run_carnival(cn, input_nodes, output_nodes, Gp, 
                          betaWeight=edge_penalty, solver=solver,
                          node_list=node_list,
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