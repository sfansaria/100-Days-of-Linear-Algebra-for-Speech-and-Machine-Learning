'''
Maximum Likelihood Estimation (MLE)

In speech ml, MLE is the foundational framework which is used to answer the critical question: 
" Given a audio recording of a human voice, How to adjust the model's weights so that the 
correct text transcription has the absolute highest probability of being generated?"
Instead of calculating the physical distance error like Mean Squared Error (MSE), MLE models the 
data as a probability distribution and treats the model weights as dials that reshapes that 
distribution.


Let us consider a speech classification setup. If the model processes 3 consecutive audio frames,
it calculates the independent probability of the true matching phonemes.
P(frame_1) = 0.7
P(frame_2) = 0.6
P(frame_3) = 0.8

To find the collective likelihood (L) of the entire speech sequence being correct,
probability theory states to multiply the individual probabilities together:
L(W) = 0.7 * 0.6 * 0.8

The problem - if we consider a audio of 10 secs, then this may have 1000 frames and multiplying 1000 decimal 
values will cause the number to shrink exponentially until it ends at (10)**-50. 
A computer cannot store a tiny number like that and it rounds it down to a flat 0.0.

The solution - Wrap the entire likelihood function in a natural logarithm (log).
Logarithms possess a unique algebraic property, they convert the multiplications into simple additions.

log(a*b) = log(a) + log(b)

By taking the Log-Likelihood, the sequence calculation transforms from a dangerous chain of multiplications
into a highly stable sum of logs: log L(W) = log(0.7) + log(0.6) + log(0.8)

Standard Optimization Algorithms are designed to walk down a mountain to minimize an error, machine learning 
software flips the sign of this value. This gives the standard speech loss function: Negative Log-Likelihood (NLL).
Minimizing this negative log-likelihood is mathematically identical to maximizing the true probability.

Lets simulate MLE by optimizing a probability distribution to match a human speaker's pitch distribution.
Lets observe how calculus changes the parameters of a Gaussian distribution (μ and σ)  
to maximize the acoustic likelihood.
'''
import torch
import torch.optim as optim
import math

torch.manual_seed(42)

#1. simulate the real human speech data (recorded and extracted their fundamental vocal pitch), original human pitch = 150.0Hz
true_pitch_data = torch.normal(mean=150.0, std=10.0, size=(50,))

#2. Initialize the model's probabilistic dials
#The model doesn't know the true mean or standard deviation. Start with random guesses.
# Turn on requires_grad because Calculus is used to optimize these statistical dials.
#guess mean
model_mu = torch.tensor([100.0], requires_grad=True)
#guess variance
model_sigma = torch.tensor([30.0], requires_grad=True)

#Optimizer to adjust the statistical dials
optimizer = optim.AdamW([model_mu, model_sigma], lr=0.5)

print("Running the Maximum Likelihood Estimation (MLE) Optimization")
print(f"Intial Model State : Guess Mean: {model_mu.item():.2f}Hz, Guess Variance: {model_sigma.item():.2f}")

for step in range(4):
    optimizer.zero_grad()

    # 3. The Gaussian Negative Log-Likelihood Formula 
    # For a normal distribution, the log-probability of observing a data point 'x' is:
    # log_p = -0.5 * log(2 * pi * sigma^2) - ((x - mu)^2 / (2 * sigma^2))
    # sum this across all 50 audio frames to get the total log-likelihood.
    variance = model_sigma**2 + 1e-6 # 1e-6 avoids division by zero

    #mathematical expansion of the log-likelihood function
    term1 = -0.5 * torch.log(2*math.pi*variance)
    term2 = -((true_pitch_data-model_mu)**2) / (2*variance)
    total_log_likelihood = torch.sum(term1+term2)

    # Flip the sign to get Negative Log-Likelihood (NLL) to minimize it
    negative_log_likelihood = -total_log_likelihood
    
    #4. calculus and optimization 
    negative_log_likelihood.backward()
    optimizer.step()

     # Clamp sigma to ensure standard deviation never goes negative
    with torch.no_grad():
       model_sigma.clamp_(min=1e-3)
    
    print(f"Step {step+1}:")
    print(f"  -> Current Negative Log-Likelihood Error: {negative_log_likelihood.item():.2f}")
    print(f"  -> Reshaped Dials -> Mean: {model_mu.item():.2f}Hz, Std: {model_sigma.item():.2f}Hz")

print("The calculus gradients successfully pushed the peak of")
print("probability distribution directly over the human audio data points")