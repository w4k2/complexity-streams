import numpy as np
from strlearn.streams import StreamGenerator
from time import time
from config import *

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

                start = time()
                X, y = StreamGenerator(**cc)._make_classification()
                print(replication, n_features, drift_type, n_clusters_per_class, X.shape, y.shape, time()-start)
                
                filename = '%s_f%i_c%i_r%i' % (drift_type, n_features, n_clusters_per_class, replication)
                print(filename)
                
                np.savez('streams/%s' % filename, X=X, y=y)
                
                X, y = None, None