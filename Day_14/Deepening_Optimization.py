'''
The mathematical evolution from Adam to AdamW

The core problem with standard Adam :
In classic gradient descent, L2 Regularization (Wright Decay) is used to keep the model weights small,
Which prevents overfitting. Mathematically, we add a penalty to the loss function based on the size of the weights.
           Loss_total = Loss_speech + (λ//2)*(||W||**2)
When you take the derivative (gradient of this combined loss), the weight decay term simply adds a small fraction 
of the current weight (λW) directly to the gradient.

Adam adaptively scales the gradient step by dividing it by the historical average of the past gradients(v_t).
If a specific acoustic feature weight has historical gradients that are tiny, Adam scales the update up.
If the gradients are huge, Adam scales them down.

The Problem - Standard Adam bundles the weight decay penalty inside the gradient calculation,
it accidently scales the penalty too! If a weight has small gradients, Adam aggresively over-decays that 
weight, erasing vital phonetic memory.

The Solution - In 2017, researchers Loshchilov and Hutter fixed this by completely decoupling the weight decay 
from the gradient step. Instead of letting the adaptive optimizer alter the regularization penalty, 
AdamW updates the weights like this:
               W_(t+1) = W_t - n_t ((m_t//(sqrroot(v_t)+epsilon)) + lambda * W_t)

By pulling the lambda*W_t outside of the adaptive fraction, the weight decay is applied uniformly regardless of
how chaotic or quiet the acoustic gradient history is. This mathematical decoupling is what allows deep Transformers 
(like Whisper or Conformer models) to remain structurally stable over long training runs.

'''

#Lets simulate an acoustic layer and track exactly how AdamW manages the scaling 
#of a parameter weight compared to raw Gradient Descent.

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#simulate 1 acoustic feature mapping to 1 phonetic output
#explicitly create a parameter tensor to watch it update
weight = torch.tensor([2.0], requires_grad=True)
bias = torch.tensor([0.5], requires_grad=True)

#Define a single target mapping (simulating audio target)
x_audio = torch.tensor([1.5])
y_target = torch.tensor([0.0])

#configuring AdamW with clear hyper-parameters
learning_rate = 0.1
weight_decay_val = 0.05
optimizer = optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay_val)

print("Track decoupled weight decay (AdamW) step by step")


for step in range(3):
    #Raw weight before this step
    w_before = weight.item()

    #simple MSE loss calculation
    prediction = weight * x_audio + bias
    loss = 0.5 * (prediction-y_target)**2

    #Calculate gradients (Calculus phase)
    optimizer.zero_grad()

    loss.backward()

    #Read the raw gradient calculated via the Chain Rule

    raw_grad = weight.grad.item()

    #Execute the optimization step (Updates the parameter based on the AdamW rules)
    optimizer.step()

    w_after = weight.item()
    delta_w = w_after-w_before
    
    print("-"*65)
    print(f"Step number : {step+1}:")
    print("-"*65)
    print(f"Loss: {loss.item():.4f}")
    print(f"Calculated Grad: {raw_grad:.4f}")
    print(f"Weight Shift: {delta_w:.4f} (Moves from {w_before:.4f} to {w_after:.4f})")
