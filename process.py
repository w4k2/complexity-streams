import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import convolve1d
from tqdm import tqdm
from scipy import stats
from torch import mean
np.set_printoptions(
    precision=3,
    suppress=True
)

# Select the solution file
filename = 'bal_sudden_f32_c2_r0'
print('# Processing the %s.' % filename)

# Load complexities [chunk_id, measure_id], measures [measure_id] and time [chunk_id]
data = np.load('complexities/%s.npz' % filename)
complexities, measures, times = [data[k] for k in ['complexities', 'measures','times']]

# Gather the basic info
n_chunks, n_measures = complexities.shape
print('# %i chunks with %i measures' % (n_chunks, n_measures))

# Define the processing parameters
alpha = .05
treshold = 2
immobilizer = 3 # minimal 3
norm_mean = np.zeros(n_measures)
norm_std = np.ones(n_measures)

# 
normalizer = [[] for measure in measures] # norm source storage
activator = [] # activity storage
isno = [] # poprawa semantyki zmiennej!!!!

RS = []
drifts = []

"""
Do the main processing loop
"""
for chunk_id, complexity_vector in enumerate(tqdm(complexities)):
    # Gather prior activity and prior normalty
    is_active = np.zeros(n_measures).astype(bool) if len(activator) == 0 else np.copy(activator[-1])
    is_normal = np.ones(n_measures) if len(isno) == 0 else np.copy(isno[-1]) # poprawa semantyki zmiennej!!!!
    
    # Gather the complexities to normalizer
    for measure_id, score in enumerate(complexity_vector):
        # Accumulate score only if non-active
        if not is_active[measure_id]:
            normalizer[measure_id].append(score)
    
    # Verify normalty
    for measure_id, input_vector in enumerate(normalizer):
        # Verify if there are at least3 samples per analysis:
        if len(input_vector) >= immobilizer:
            # Calculate p-value only if the measure is non-active
            is_normal[measure_id] = stats.shapiro(input_vector).pvalue if not is_active[measure_id] else is_normal[measure_id]
            
            # Verify if to activate measure
            p = is_normal[measure_id]

            # Verify if just activated
            if p < alpha and (not is_active[measure_id]):
                #print('JUST ACTIVATED %i [%s] at %i' % (measure_id, measures[measure_id], chunk_id))
                norm_mean[measure_id] = np.mean(input_vector)
                norm_std[measure_id] = np.std(input_vector)

            # Store activation fact
            if p < alpha:
                is_active[measure_id] = True
                
    # Drift detection
    is_drift = False

    # If any detector is active
    if np.sum(is_active) > 0:
        # Get all the norm info of active detectors        
        rescaled = np.abs((complexity_vector - norm_mean) / norm_std)[is_active]
        
        # Integrate and threshold it
        r = np.median(rescaled)        
        if r > treshold:
            is_drift = True
            
        RS.append(r)
        
    # Drift reset
    if is_drift:
        # Reset normalizer
        normalizer = [[] for measure in measures]
        
        # Deactivate detectors
        is_active = np.zeros(n_measures).astype(bool)        
                
    # Store info
    drifts.append(is_drift)
    activator.append(is_active)
    isno.append(is_normal)
    
"""
Presentation
"""
isno = np.array(isno)
activator = np.array(activator)

bw = 15
fig, ax = plt.subplots(2,2,figsize=(bw, bw))

for m_idx, row in enumerate(isno.T):
    ax[0,0].plot(row, label=measures[m_idx])
ax[0,0].legend()
ax[0,0].set_title('P-values of measure normalty')

ax[0,1].plot(drifts)
ax[0,1].set_title('Detected drifts')

ax[1,1].plot(RS)
ax[1,1].set_title('R-vector')

for m_idx, row in enumerate(activator.T):
    ax[1,0].plot(row, label=measures[m_idx])
ax[1,0].legend()
ax[1,0].set_title('Measure activity')

plt.tight_layout()
plt.savefig('foo.png')