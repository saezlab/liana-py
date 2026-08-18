from liana.method import build_prior_network, find_causalnet

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
                                     solver='scipy',
                                     seed=1337,
                                     max_runs=50,
                                     stable_runs=20,
                                     )

    assert problem.weights == [1.0, 0.01, 1.0, 1.0]
    assert (df_res[df_res['target_type']=='output']['target'].isin(['M1', 'M2'])).all()



def test_causalnet_noweights():
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=False)
    df_res, problem = find_causalnet(prior_graph,
                                     input_scores,
                                     output_scores,
                                     node_weights={"N1": 1, "N2": 0},
                                     verbose=False,
                                     solver='scipy',
                                     max_runs=50,
                                     stable_runs=20,
                                     )
    assert df_res['source_pred_val'].values.sum() == 8
    assert df_res['target_pred_val'].values.sum() == 9
