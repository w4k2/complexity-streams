import numpy as np
from strlearn.streams import StreamGenerator
from config import *
from tqdm import tqdm

pbar = tqdm(total=len(replications)*len(dimensionalities)*len(drift_types)*len(number_of_clusters))

for replication, random_state in enumerate(replications):
    for n_features in dimensionalities:
        for drift_type in drift_types:
            for n_clusters_per_class in number_of_clusters:    
                cc = {
                    **static,
                    **drift_types[drift_type],
                    'n_features': n_features,
                    'n_informative': n_features,
                    'n_clusters_per_class': n_clusters_per_class,
                    'random_state': random_state
                }

                X, y = StreamGenerator(**cc)._make_classification()
                pbar.update()
                
                filename = '%s_f%i_c%i_r%i' % (drift_type, n_features, n_clusters_per_class, replication)
                
                np.savez('streams/%s' % filename, X=X, y=y)
                
                X, y = None, None