from liana.method import find_causalnet, build_prior_network

input_pkn = [
    ("I1", 1, "N1"),
    ("N1", 1, "M1"),
    ("N1", 1, "M2"),
    ("I2", -1, "N2"),
    ("N2", -1, "M2"),
    ("N2", -1, "M1"),
    ]

input_scores = {"I1": 1, "I2": 1}
output_scores = {"M1": 1, "M2": 1}
node_weights = {"N1": 1, "N2": 1}

def test_build_prior_network():
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=True)
    assert prior_graph.num_vertices == 6
    assert prior_graph.num_edges == 6
    

def test_caulsalnet():
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=False)
    df_res, problem = find_causalnet(prior_graph, 
                                     input_scores, 
                                     output_scores, 
                                     node_weights=node_weights,
                                     verbose=False,
                                     show_solver_output=False
                                     )

    assert problem.weights == [1.0, 0.01, 1.0]
    assert df_res['source_pred_val'].values.sum() == 5
    assert df_res['target_pred_val'].values.sum() == 8
    assert df_res[df_res['source_type']=='input']['source'].values[0] == 'I2'
    assert (df_res[df_res['target_type']=='output']['target'].isin(['M1', 'M2'])).all()
    


def test_causalnet_noweights():
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=False)
    df_res, problem = find_causalnet(prior_graph,
                                     input_scores,
                                     output_scores,
                                     node_weights={"N1": 1, "N2": 0},
                                     verbose=False,
                                     show_solver_output=False
                                     )
    assert df_res['source_pred_val'].values.sum() == 9
    assert df_res['target_pred_val'].values.sum() == 10
