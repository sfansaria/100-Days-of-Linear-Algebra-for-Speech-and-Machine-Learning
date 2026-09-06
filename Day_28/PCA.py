'''
Principal Component Analysis (PCA)
SVD is an abstract matrix factorization tool, whereas PCA is the practical statistical application of that math used for 
Dimensionality Reduction.

lets see this through speech, imagine a speech model is fed an audio window with 80 Mel-sepctrogram frequency channel. 
Processing all 80 features frame by frame is computationally expensive. When a human speaks, many frequency channels move 
together in lockstep. (If frequency channel 12 goes up, then frequency channel 13 also goes up because they share the same 
vowel harmonic). Keeping both the channels as separate numbers is redundant.
PCA acts like an automated audio spotlight engineer. It calculates the Covariance Matrix of the data to observe how all the 
channels relate to each other. Then, it rotates the coordinates system, creating new, synthetic axes called Principal Components.

component 1 - points in the direction of the absolute largest acoustic variance, capturing the primary vocal volume shifts.
component 2 - captures the next largest independent shift, the vowel tone adjustments.

By keeping only the top 3 or 4 Principal Components, compress the 80-channel feature grid down to a tight, lighweight vector,
stripping away redundant calculations while preserving the primary phonetic variance.

--- Take the acoustic feature matrix,
--- calculate its mean-centered covariance matrix, 
--- extract its principal coordinate directions,
--- reduce its dimension from 20 channels down to 2.


'''


import torch
torch.manual_seed(42)


X_acoustics_features = torch.randn(100, 20) #100 frames of 20-channel acoustic data

#PCA requires data to be centered around the spatial origin (0)
X_centered = X_acoustics_features - torch.mean(X_acoustics_features, dim=0) #Mean Centering 

#computing the covariance matrix
#Formula - (X^T * X)/(N-1)
#Shape will be mapping how each frequency channel covaries with others
N = X_centered.shape[0]
covariance_matrix = torch.matmul(X_centered.t(), X_centered) / (N-1)

#Eigen-Decomposition (Core PCA)
#extract eigen vectors (directions) and eigenvalues (important scores)
eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

#PyTorch returns them sorted ascendingly, so reverse to get higher variance first
eigenvalues = torch.flip(eigenvalues, dims=[0])
eigenvectors = torch.flip(eigenvectors, dims=[1])


#Project Data onto Lower Dimensional Space
#select the top 2 Principal Component Projection vectors
top_2_components = eigenvectors[:, :2] #shape : [20, 2]

#Project original 20-dimensional audio down to a tight 2-dimensional  vector matrix
compressed_pca_features = torch.matmul(X_centered, top_2_components)

print("Principal Component Analysis (PCA) Dimensionality Reduction")
print(f"Original High-Dim Audio Matrix Feature Shape: {list(X_acoustics_features.shape)}")
print(f"Covariance Matrix Map Structural Shape: {list(covariance_matrix.shape)}")
print(f"Top 2 Eigenvalue Variance Coverage Scores: {eigenvalues[0].item():.2f}, {eigenvalues[1].item():.2f}")
print(f"Final Compressed PCA Feature Matrix Shape: {list(compressed_pca_features.shape)}")