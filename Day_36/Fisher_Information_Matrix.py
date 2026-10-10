'''
Fisher information Matrix-
 When a speech model is trained, changing a weight by a tiny increment of +0.01 can
 have wildly different real-world impacts depending on where that weight sits.

 - For one parameter, a +0.01 change might barely shift the model's output at all.
 - For another parameter, a +0.01 change might completely scramble the model's confidence, 
 causing it to misclassify vowels.

Stancalculus optimization (like AdamW or SGD) is completely blind to this. It assumes all
directions on the weight grid are flat and equal.

The Fisher Information Matrix (FIM) is a specialized cousin of the Hessian. Instead of 
measuring the curvature of an arbitrary loss function, it measures the local curvature 
of the probability distribution space itself. 

Mathematically, it is the expected value of the outer product of the log-likelihood gradients:
                F = E[delta_theta log P(x|theta) delta_theta log P(x|theta)_T]

Information Theory proves that the Fisher Information Matrix is exactly equal to the Hessian of the KL Divergence.
It maps out the local metric tensor of how fast information is lost or distorted when the weight's of the model is tweaked.

The Application of Fisher Information Matrix: Natural Gradient Descent

The true probability space of the model is curved like a globe, taking a standard straight step (Standard Gradient Descent)
across the flat parameter grid creates a massive distortion error frequently causing the  model to overcorrect, get stuck in
plateaus or undergo catatrophic forgetting.

Natural Gradient Descent corrects this by multiplying the standard gradient vector by the inverse of the Fisher Information Matrix.

theta_t+1 = theta_t - n * F^-1 delta_theta L

Its forces the optimization algorithm to take steps of constant length in terms of KL Divergence (true information change)
rather than arbitrary coordinate length

Speech distributions suffer from massive imbalances (example common words vs rare regional accents).
Natural Gradient ensures that adapting a speech encoder to a rare accent takes a safe step that does 
not inadvertently warp or erase the model's fundamental linguistic distribution rules.

'''

import torch 
import torch.nn as nn

torch.manual_seed(42)


# STEP 1: Define a Simple Probabilistic Speech Distribution
# Imagine a 1D model predicting a speaker's single vocal formant coordinate.
# The distribution is parameterized by a mean (mu) and variance (sigma).
mu = torch.tensor([120.0], requires_grad=True)
sigma = torch.tensor([15.0], requires_grad=True)


#simulate receiving a batch of real voice observations
voice_samples = torch.normal(mean=120.0, std = 15.0, size=(100,))

print("Running the Fisher Information Matrix (FIM) Calculus")

# STEP 2: Calculate Log-Likelihood Gradients (Scores)
# To build the Fisher Matrix, 
# extract how the log-probability of the speech data changes with respect to EACH parameters independently.

variance = sigma ** 2 + 1e-6
log_likelihood = -0.5 * torch.log(2 * 3.14159 * variance) - ((voice_samples - mu)**2) / (2 * variance)

#Initialize placeholders to stack gradients for each sample
grad_mu_list = []
grad_sigma_list = []

for sample_idx in range(len(voice_samples)):
    # Isolate the log-likelihood of a single individual sound observation
    single_log_p = log_likelihood[sample_idx]
    
    #calculate first-order derivatives (Clacu;us phase)
    grads = torch.autograd.grad(single_log_p, [mu, sigma], retain_graph=True)

    grad_mu_list.append(grads[0])
    grad_sigma_list.append(grads[1])


# Stack into cohesive gradient track vectors: Shape [100 x 1]
grad_mu_vector = torch.stack(grad_mu_list)
grad_sigma_vector = torch.stack(grad_sigma_list)



# STEP 3: CONSTRUCT THE FISHER INFORMATION MATRIX
# The Fisher matrix elements are the expected values (averages) of the outer products of these specific log-likelihood gradients.
fim_00 = torch.mean(grad_mu_vector * grad_mu_vector)       # Curvature of Mean vs Mean
fim_01 = torch.mean(grad_mu_vector * grad_sigma_vector)    # Cross-interaction
fim_10 = fim_01                                            # Symmetric property
fim_11 = torch.mean(grad_sigma_vector * grad_sigma_vector) # Curvature of Std vs Std

# Assemble the 2x2 Information Geometry Metric Tensor Matrix
fisher_information_matrix = torch.tensor([
                                          [fim_00.item(), fim_01.item()],
                                          [fim_10.item(), fim_11.item()]
                                        ])


# STEP 4: DIAGNOSTIC METRIC OUTPUTS
print(f"1. Extracted Gradient Vector Dim: {list(grad_mu_vector.shape)} across 100 audio samples")
print("-"*75)
print("2. COMPUTED FISHER INFORMATION MATRIX (Metric Tensor Map):")
print(fisher_information_matrix.numpy())
print("-"*75)
print(f"   -> Mu-Axis Curvature Sensitivity:  {fisher_information_matrix[0,0].item():.6f}")
print(f"   -> Sigma-Axis Curvature Sensitivity: {fisher_information_matrix[1,1].item():.6f}")
print("   -> Interpretation: This matrix acts as the geometric radar mapping the")
print("      curved topography of your model's probability landscape.")
print("="*75)