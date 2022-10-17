import numpy as np

filename = 'bal_incremental_f32_c4_r0'
print(filename)

data = np.load('streams/%s.npz' % filename)
print(data)
print(data.files)

X, y = data['X'], data['y']

print(X.shape, y.shape)