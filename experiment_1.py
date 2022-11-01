from sklearn.naive_bayes import GaussianNB
from config import *
from tqdm import tqdm
import numpy as np
from detectors.DDM import DDM
from detectors.EDDM import EDDM
from detectors.ADWIN import ADWIN
from detectors.HDDM_AA import HDDM_AA
from detectors.HDDM_WW import HDDM_WW
from detectors.CDDE import CDDE
from detectors.meta import Meta
import problexity as px

# Establish base stream info
n_chunks = static['n_chunks']
chunk_size = static['chunk_size']
n_clusters_per_class = number_of_clusters[0]
n_detectors = 6

measures = [getattr(px.classification, n) 
            for n in px.classification.__all__]
metric_filter = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
measures = measures[metric_filter]

# Prepare storage for complexities and time
detection_results = np.zeros((len(replications), len(dimensionalities), len(drift_types), n_detectors, n_chunks-1))

# Do the main loop
i = 0 # counter
for replication, random_state in enumerate(replications):
    for n_features_id, n_features in enumerate(dimensionalities):
        for drift_type_id, drift_type in enumerate(drift_types):
            
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
                data = np.load('/Volumes/T7/ComplexityStreams/%s.npz' % filename)
            except:
                print('NO FILE YET')
                exit()
            X, y = data['X'], data['y']
            print(X.shape, y.shape)
            
            # Define detectors
            detectors = [
                Meta(base_clf=GaussianNB(), detector=DDM()),
                Meta(base_clf=GaussianNB(), detector=EDDM()),
                Meta(base_clf=GaussianNB(), detector=ADWIN()),
                Meta(base_clf=GaussianNB(), detector=HDDM_AA()),
                Meta(base_clf=GaussianNB(), detector=HDDM_WW()),
                Meta(base_clf=GaussianNB(), detector=CDDE(measures=measures))
            ]       
            
            # Iterate stream
            for chunk_id in tqdm(range(n_chunks)):
                a, b = chunk_id*chunk_size, (chunk_id+1)*chunk_size
                _X, _y = X[a:b], y[a:b]

                for detector in detectors:
                    # Test then train
                    try:
                        detector.predict(_X)
                    except:
                        pass
                    detector.partial_fit(_X, _y)
            
            for detector_id, detector in enumerate(detectors):
                detection_results[replication, n_features_id, drift_type_id, detector_id] = detector.detector.drift
                print(detector.detector.drift)
                # exit()
            
            # Store results
            np.save('results/exp_comparison', detection_results)
        
