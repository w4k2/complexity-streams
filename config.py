np.random.seed(1410)
static = {
    'n_chunks': 100, 'chunk_size': 100,
    #'n_chunks': 2000, 'chunk_size': 500,
    'n_drifts': 7,
    'n_redundant': 0, 'n_repeated': 0
}

dimensionalities = [8,16,32]
drift_types = {
    'bal_sudden': {'concept_sigmoid_spacing':999},
    'bal_gradual': {'concept_sigmoid_spacing':5},
    'bal_incremental': {'concept_sigmoid_spacing':5, 'incremental':True},
    'imb_sudden': {'concept_sigmoid_spacing':999, 'weights': (static['n_drifts'], 10, .7)},
    'imb_gradual': {'concept_sigmoid_spacing':5, 'weights': (static['n_drifts'], 5, .7)},
    'imb_incremental': {'concept_sigmoid_spacing':5, 'incremental':True, 'weights': (static['n_drifts'], 5, .7)},
}
number_of_clusters = [2,3,4]
replications = np.random.randint(9999, size=10)