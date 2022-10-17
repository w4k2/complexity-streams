from config import *
from tqdm import tqdm
import problexity as px
import numpy as np

# Get all the classification measures
measures = [getattr(px.classification, n) 
            for n in px.classification.__all__]
n_measures = len(measures)
print(px.classification.__all__)
print(measures, n_measures)

# Establish base stream info
n_chunks = 10 # static['n_chunks']
chunk_size = static['chunk_size']

# Do the main loop
i = 0 # counter
for replication, random_state in enumerate(replications):
    for n_features in dimensionalities:
        for drift_type in drift_types:
            for n_clusters_per_class in number_of_clusters:
                # Establish filename
                filename = '%s_f%i_c%i_r%i' % (
                    drift_type, n_features, 
                    n_clusters_per_class, replication
                )
                # Say the counter
                print('stream %i - %s' % (i, filename))
                i += 1
                
                # Load the data
                try:
                    data = np.load('streams/%s.npz' % filename)
                except:
                    print('NO FILE YET')
                    exit()
                X, y = data['X'], data['y']
                print(X.shape, y.shape)
                
                # Prepare storage for complexities
                complexities = np.zeros((n_chunks, n_measures))
                
                # Iterate stream
                for chunk_id in tqdm(range(n_chunks)):
                    a, b = chunk_id*chunk_size, (chunk_id+1)*chunk_size
                    _X, _y = X[a:b], y[a:b]
                
                    complexities[chunk_id] = [measure(_X, _y) for measure in measures]
                    
                # Store results
                np.savez('complexities/%s' % filename,
                         complexities=complexities,
                         measures=px.classification.__all__)
            
