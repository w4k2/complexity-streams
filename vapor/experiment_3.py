from torch import threshold
import strlearn as sl
import numpy as np
import problexity as px
from tqdm import tqdm
from detectors.meta import Meta
from detectors.CDDE import CDDE
from sklearn.naive_bayes import GaussianNB

def noise_attributes(stream, scale):
    
    X_np, y_np = stream._make_classification()
    # print(X_np.shape)

    features_std = np.std(X_np, axis=0)
    # print(features_std.shape)

    noise = np.random.normal(0,scale*features_std,X_np.shape)
    # print(noise)

    X_np = X_np+noise
    
    file = np.concatenate([X_np, y_np[:,np.newaxis]], axis=1)
    np.save('stream_generated.npy', file)
    np.savetxt('stream_generated.txt', file)

    s = sl.streams.NPYParser("stream_generated.npy", chunk_size=stream_static['chunk_size'], n_chunks=stream_static['n_chunks'])
    return s

measures = np.array([getattr(px.classification, n) 
            for n in px.classification.__all__])
metric_mask = np.ones_like(measures).astype(bool)
metric_mask[4] = False
measures = measures[metric_mask]

# streams
stream_static = {
        'n_drifts': 2,
        'n_chunks': 150,
        'chunk_size': 250,
        'n_features': 10,
        'n_informative': 10,
        'n_redundant': 0,
        'recurring': False,
        'weights': (2, 999, .7)
    }

# setup
np.random.seed(9038984)

reps = 3
random_states = np.random.randint(100,10000,reps)

y_flip = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
attr_noise = [0.0] #np.linspace(0,1,5)

t = reps*len(y_flip)*len(attr_noise)
pbar = tqdm(total=t)

results_clf = np.zeros((len(y_flip), len(attr_noise), reps, stream_static['n_chunks']-1))
results_drf_arrs = np.zeros((len(y_flip), len(attr_noise), reps, stream_static['n_chunks']-1))

for flip_id, flip in enumerate(y_flip):
    for attr_n_id, attr_n in enumerate(attr_noise):
        for replication in range(reps):

            detectors = [
                Meta(GaussianNB(), CDDE(measures = measures, treshold=0.75)),
            ]
            
            config = {
                **stream_static,
                'random_state': random_states[replication],
                'y_flip':flip
                }

            stream = sl.streams.StreamGenerator(**config)

            stream = noise_attributes(stream, attr_n)

            print("replication: %i, attr_noise: %f, yflip: %f" % (replication, attr_n, flip))

            eval = sl.evaluators.TestThenTrain()
            eval.process(stream, detectors)

            scores = eval.scores
            results_clf[flip_id, attr_n_id, replication] = scores[:,:,0]

            results_drf_arrs[flip_id, attr_n_id, replication] = np.array(detectors[0].detector.drift)
            print(np.argwhere(results_drf_arrs[flip_id, attr_n_id, replication]==2).flatten())

            pbar.update(1)

np.save('results/exp3', results_clf)
np.save('results/exp3_drf', results_drf_arrs)

pbar.close()