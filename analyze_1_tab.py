from config import *
import numpy as np
import matplotlib.pyplot as plt
from methods import dderror, find_real_drift
from tabulate import tabulate
from scipy.stats import ttest_rel

# Establish base stream info
n_chunks = static['n_chunks']
chunk_size = static['chunk_size']
n_clusters_per_class = number_of_clusters[0]
detectors = ['DDM', 'ADWIN', 'HDDM_A', 'HDDM_W', 'CDDE']
drifts = find_real_drift(2000, 7)

# Load results
detection_results = np.load('results/exp_comparison_all.npy')
#(replications, dimensionalities, drift_types, n_detectors, n_chunks-1)

errors = np.zeros((len(dimensionalities), len(drift_types), len(detectors), 10, 3))

for dim_id, dim in enumerate(dimensionalities):
    for dt_id, dt in enumerate(drift_types):
        r = detection_results[:,dim_id,dt_id]
        for d in range(len(detectors)):
            for rep in range(10):
                detections = np.argwhere(r[rep,d]==2).flatten()
                errors[dim_id, dt_id, d, rep] = dderror(drifts, detections, n_chunks)

                
alpha = 0.05

for metric_id in range(3):
        
    t = []
    for dim_id, dim in enumerate(dimensionalities):
        for dt_id, dt in enumerate(drift_types):
            row = "dim %i %s" % (dim, dt)
                
            res_temp = errors[dim_id, dt_id, :, :,metric_id]
            res_temp = res_temp.swapaxes(0,1)
            e_res_temp = np.mean(res_temp, axis=0)
            std_res_temp = np.std(res_temp, axis=0)
            
            
            length = len(e_res_temp)

            s = np.zeros((length, length))
            p = np.zeros((length, length))

            for i in range(length):
                for j in range(length):
                    s[i, j], p[i, j] = ttest_rel(
                        res_temp.T[i], res_temp.T[j])
                    
            _ = np.where((p < alpha) * (s < 0))

            conclusions = [list(1 + _[1][_[0] == i])
                            for i in range(length)]

            t.append(["%s" % row] + ["%.3f" % v for v in e_res_temp])
            # t.append([''] + ["%.3f" % v for v in std_res_temp])

            t.append([''] + [", ".join(["%i" % i for i in c])
                                if len(c) > 0 and len(c) < length-1 else ("all" if len(c) == length-1 else "---")
                                for c in conclusions])

        print(tabulate(t))
        
    with open('tables/e1_m%i.txt' % metric_id, 'w') as f:
        f.write(tabulate(t, detectors, floatfmt="%.3f", tablefmt="latex"))