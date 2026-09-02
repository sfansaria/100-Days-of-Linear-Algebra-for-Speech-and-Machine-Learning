'''
when RNN-T or CTC model outputs the massive sequential matrix of phoneme probabilities,
how do we measure if the model is truly learning or just blindly guessing? 
We need specific metrics to evaluate the model's performance and learning capability.

1. Entropy: This measures the uncertainty in the model's predictions.
If the model is inspecting an audio frame conatining a clean vowel sound like "/aa/".
If the model outputs a probability of 0.99 for "/aa/" and 0.01 for all other phonemes,
then the model is confident and has low entropy means low confusion and high certainty in its predictions.

But if the audio conatains background noise or a complex sound, then the model gets confused and outputs a 
flat 0.25 probability for four different phonemes, then the model is confused what is is hearing and 
has high entropy and high uncertainty in its predictions.

Entropy calculates the average uncertainty in a single probability distribution vector P. 
H(P) = - SUM_i(P(X_i)log P(X_i))

In Speech ML, Cross-Entropy Loss and Kullback-Leibler Divergence, dictates how an optimization algorithm 
reshapes a network's raw predictions to match the true human speech patterns.

2. Cross-Entropy Loss: This measures the total average "surprise" or informational cost when the model predicts
events using an incorrect probability distribution (Q) isntead of the true traget distribution (P).

Mathematically, for a single audio frame across a vocabulary of phonemes, it is defined as:
     H(P, Q) = - (i = 1 to C) SUM(P(X_i) log(Q(X_i)))


In the classification scenario, like mapping an audio frame to characters, 
the true target distribution P is One-Hot vector. If the correct letter is "A" (index 0 out of 3 choices),
then the vector is precisely: P[1.0, 0.0, 0.0] 
When the summation of the cross-entropy is expanded for this frame, the 0.0 values erase the wrong categories.
H(P, Q) = -[1.0 . log Q(x_0) + 0.0 . log Q(x_1) + 0.0 . log Q(x_2)]
H(P, Q) = - log Q(x_0)

This is why Cross-Entropy is practically implemented as Negative Log Likelihood (NLL). 
The optimizer ignores what the model predicted for the wrong answers and focuses entirely on 
computing gradients to force the single correct class probability (Q(x_0)) to shoot up toward 1.0.

3. Kullback-Leibler Divergence: This measures the statistical "distance" or information loss between two continuous, 
fluid probability distributions.
Mathematically it is defined as: 
D_KL(P || Q) = (i = 1 to C) Summation(P(x_i) log (P(x_i)//Q(x_i)))
             = (i = 1 to C) Summation(P(x_i)(log(P(x_i)) - log(Q(x_i))))
             = (i = 1 to C) Summation(P(x_i)(log(P(x_i))) -  (i = 1 to C) Summation(P(x_i)(log(Q(x_i))) 
             = - Entropy(P) + Cross-Entropy(P, Q)

Cross-Entropy(P, Q) = Entropy(P) + D_KL(P || Q)


The KL Divergence is asymmetric, meaning (D_KL(P || Q) != (D_KL(Q || P)).
It matters immensely which distribution to put first. If P(x) > 0 but the model predicts Q(x) = 0, 
the fraction P(x)//Q(x) hits a division-by-zero, causing the KL error to explode to infinity.

If training a generative voice model using D_KL(P || Q)), the model is heavily penalized 
if it misses any part of the real human voice distribution. 
It forces the model to be zero-avoiding, making its voice outputs wide, smooth, and sometimes 
slightly blurry to remain mathematically safe.

When do to use which?  

- Cross-Entropy is used when there is a absolute, unambiguous ground-truth choice.
Training the final text token prediction layers in CTC Loss or RNN-Transducers.
The audio frame either represents the letter "t" or it doesn't. 
Cross-Entropy forces the model to draw sharp, distinct classification boundaries.

- KL Divergence is used when for manipulating complex, continuous target landscapes 
  (like matching human vocal character, accent styles, or pitch profiles).
  
  - Knowledge Distillation: it used to compress a massive, state-of-the-art 
    1-billion-parameter speech encoder (Teacher) down to a lightweight 50-million-parameter
    model (Student) to run on a smartphone, you don't use hard text labels. 
    KL Divergence is used to force the Student's messy probability distribution field to 
    match the smooth, nuanced distribution output of the Teacher.
  
  - Generative Speech (Variational Autoencoders / Diffusion): In Text-to-Speech (TTS),
    KL Divergence is used to force the latent vector of a speaker's voice to match a 
    standard Gaussian bell curve prior. This guarantees that your model maps acoustic traits 
    (like breathiness or emotional tone) onto a smooth coordinate space without creating random 
    gaps or structural breaks.

'''

import torch

torch.manual_seed(42)

#The speech profile
#P represents the true speaker distribution over 3 accent categories (US, UK, AU)
# The true speaker has a mixed accent profile (Nuanced Continuous Distribution)

P_true_distribution = torch.tensor([0.70, 0.20, 0.10])

# Model A makes an excellent, smooth approximation of this accent profile
logits_A = torch.tensor([2.0, 0.8, 0.1])
Q_model_A = torch.softmax(logits_A, dim = 0)

# Model B completely misjudges the voice and misses the primary category entirely
logits_B = torch.tensor([-1.5, 2.5, 1.0])
Q_model_B = torch.softmax(logits_B, dim = 0)



def compute_metrics(P, Q):
    eps = 1e-12 #to safeguard against the log(0) computational crashes

    entropy_P = -torch.sum(P * torch.log(P+eps))
    cross_entropy = -torch.sum(P * torch.log(Q+eps))
    kl_divergence = torch.sum(P * torch.log((P+eps) / (Q+eps)))

    return entropy_P, cross_entropy, kl_divergence

#execute the calculations
ent_P, ce_A, kl_A = compute_metrics(P_true_distribution, Q_model_A)
_, ce_B, kl_B = compute_metrics(P_true_distribution, Q_model_B)

print("Research metric breakdown: Cross-entropy vs KL Divergence")
print(f"True Distribution (P): {[round(x, 3) for x in P_true_distribution.tolist()]}")
print(f"Model A Prediction (Q_A): {[round(x, 3) for x in Q_model_A.tolist()]}")
print(f"Model B Prediction (Q_B): {[round(x, 3) for x in Q_model_B.tolist()]}")
print(f"Base Information Entropy (H(P)) of the Source: {ent_P.item():.4f}")
print("Model A Diagnostics (Accurate Alignment):")
print(f"Cross-Entropy Loss H(P, Q_A): {ce_A.item():.4f}")
print(f"KL Divergence D_KL(P||Q_A): {kl_A.item():.4f}")
print(f"Verification Identity Check: {ent_P.item() + kl_A.item():.4f}") #must be equal to cross - entropy

print("Model B Diagnostics (Accurate Alignment):")
print(f"Cross-Entropy Loss H(P, Q_B): {ce_B.item():.4f}")
print(f"KL Divergence D_KL(P||Q_B): {kl_B.item():.4f}")