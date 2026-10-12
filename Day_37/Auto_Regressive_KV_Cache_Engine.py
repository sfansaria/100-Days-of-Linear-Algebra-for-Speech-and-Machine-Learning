'''
When a speech transformer is generating text transcription token-by-token 
(auto-regressive decoding), it outputs the text sequentially:
step 1: "The"
step 2: "The cat"
step 3: "The cat sat"

The Self-Attention layer built earlier, its operation is 
Attention(Q, K, V) = softmax((QK_T)/Sqr_root(d_k))V.

The Problem:

At step 3, the model re-computes the Query, Key, and Value vectors for 
"The", "cat", and "sat". This means mathematically recalculating the 
exact same Key and Value representations for the past words over and over
again at every single new token step.

The Complexity Scaling:

For a long audio file that outputs a 500-word paragraph, this redundant 
re-calculation causes the computational complexity to spike to quadratic 
runtime levels (O(T**2)) during generation time, slowing down the real-time
transcription speeds

The Solution:

To solve this waste, engineers inject an algorithmic memory block called a KV Cache.
Instead of recomputing everything, the model calculates the Key (K) and Value (V) vectors only
for the singke brand-new word generated at the current time step.
It then takes those new vectors and appends them directly to a persistent, growing tensor register (the cache)
holding all past Keys and Values.

Step 3 Input: "sat"  ──► [ Compute Single Q, K, V Vectors ]
                                  │           │
                                  ▼           ▼
                     Append to: [ Past K ]  [ Past V ]  ◄── (The KV Cache Matrix)
                                  │           │
                                  ▼           ▼
                     Run Attention Matrix Calculations only on the fused tensors

The Mathematical Payoff: 

By bypassing the redundant historical projections, the decoding computational complexity is compressed
from a sluggish quadratic scaling (O(T**2)) down to a perfectly flat, lightning-fast linear scaling (O(T)).
This exact piece of systems engineering that allows modern voice systems to respond instantly on the screen as one speaks.

'''