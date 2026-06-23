import numpy as np
from sklearn.decomposition import PCA

# 1. SIMULATE MASSIVE DATA
# Imagine we have 100 flattened high-res images. Each has 50,000 pixels (features).
X_huge = np.random.rand(100, 50000)
print("Original Matrix Shape:", X_huge.shape) # (100, 50000)

# 2. INITIALIZE PCA
# n_components=0.95 tells PCA: "Compress my data, but stop compressing 
# when you hit the point where 95% of the original information is still preserved."
pca = PCA(n_components=0.95, random_state=42)

# 3. FIT AND TRANSFORM
# pca.fit learns the optimal math projections from the data layout.
# pca.transform actually rewrites the matrix into the compressed form.
X_compressed = pca.fit_transform(X_huge)

print("Compressed Matrix Shape:", X_compressed.shape)
# The width will drop from 50,000 down to a fraction of that size!