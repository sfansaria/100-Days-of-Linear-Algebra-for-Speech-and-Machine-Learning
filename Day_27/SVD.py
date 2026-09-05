'''
Singular Value Decomposition (SVD)-
lets imagine an audio recording of a crowded cocktail party. The microphone caprures massive, messy matrix of 
sound of waves conatining overlapping voices, background music clatter, a humming air conditioner, and clinking glasses.
SVD acts like a mathematical prism. When a chaotic light beam is passed through a prism, it splits it cleanly into
its distinct rainbow frequencies. SVD takes the messy speech matrix (A) and factorises it into three distinct matrices.

                       A = U Σ V_T
where U (Left Singular Vectors) is the Acoustic Profile that maps the unique "characters" or foundational acoustic shapes
hidden in the data (specific vocal vowel tracks vs background noise traits)

Σ (Sigma /Singular Values) is the energy knobs is a sorted list of numbers from largest to smallest. It acts as an array of volume
knobs telling exactly how loud or important each acoustic profile is. The first few numbers hold 90% of the vocal energy, the tiny 
numbers at the end represent low-value ambient static.

V_T (Right Singular Vectors / Timeline tracks) maps exactly when those specific acoustic profiles occurred over the audio timeline.

For a massive 1000 - element speech weight matrix, calculate its SVD, and intentionally erase all the tiny volume knobs 
at the bottom of the  Σ matrix, by performing a low-rank approximation. Strip away the microscopic ambient noise while
keeping the core human speech profile perfectly intact.

lets generate a simulated raw acoustc spectrogram matrix containing a clean vocal frequency corrupted by random environmental noise.
torch.liaalg.svd is used to perform the matrix factorization, isolate the core singular components, and compress the matrix.

'''

import torch

torch.manual_seed(42)


#simulate a noisy speech spectrogram matrix
time_steps, freq_bins = 30, 20

#create a clean, strutured vocal harmonic wave matrix
t = torch.linspace(0, 2 * 3.14, time_steps).unsqueeze(1)
f = torch.linspace(1, 5, freq_bins).unsqueeze(0)
clean_vocal_matrix = torch.sin(t*f)

#corrupt it with random background mic hiss/noise matrix
noise_matrix = torch.randn(time_steps, freq_bins) * 0.4

noisy_speech_matrix = clean_vocal_matrix + noise_matrix

#SVD factorisation operation
#PyTorch returns U, S (diagonal entries of Sigma), and V_T (V transpose)

U, S, V_T = torch.linalg.svd(noisy_speech_matrix, full_matrices=False)

#Truncate SVD Compression (Low-Rank Approximation)
print("Singular Value Decomposition (SVD) Compression Diagnostics")
print(f"Top 5 Singular Value Energy Knobs (S): {[round(x, 2) for x in S[:5].tolist()]}")
print(f"Bottom 3 Singular Value Energy Knobs: {[round(x, 2) for x in S[-3:].tolist()]}")


#Choose a Rank r = 2, keep only the top 2 directional vectors
rank_r = 2
U_truncated = U[:, :rank_r]
S_truncated = torch.diag(S[:rank_r])
V_T_truncated = V_T[:rank_r, :]

#Reconstruct the compressed matrix: U_truncated * S_truncated * V_T_truncated
compressed_speech_matrix = torch.matmul(torch.matmul(U_truncated, S_truncated), V_T_truncated)

#Calculated loss of data error (Frobenius norm of difference)
reconstruction_error = torch.norm(noisy_speech_matrix - compressed_speech_matrix)

print(f" Original Noisy Matrix Shape: {list(noisy_speech_matrix)}  (600 data points)")
print(f"Component Shapes -> U: {list(U_truncated.shape)}, Sigma: {list(S_truncated.shape)}, V^T: {list(V_T_truncated.shape)}")
print(f"Matrix Reconstruction Error: {reconstruction_error.item():.4f}")
print("SVD split the audio matrix and compressed its noise space")