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

lets uild a functional Auto-Regressive KV-Cache Engine. 
Simulate generating a sentence step-by-step, manage a persistent cache matrix,
and track the dimensions to see how it optimizes memory operations.
'''

import torch
import torch.nn as nn
import math

torch.manual_seed(42)

class KVCachedAttention(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        #simple projection layers for a single attention head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_new, kv_cache=None):
        """
        x_new: The single brand-new text token vector entering the model. Shape: [Batch(1), Time(1), Dim(64)]
        kv_cache: A dictionary tuple containing ('past_key', 'past_value') tensors from earlier frames
        """
        
         # 1. Project ONLY the new token into its respective Q, K, V vectors
        q_new = self.w_q(x_new) #Shape: [1, 1, 64]
        k_new = self.w_k(x_new) 
        v_new = self.w_v(x_new)


        # 2. Cache Fusion Engine Operation
        if kv_cache is None:
           # First token step: initialize the cache with our current vectors
           cached_k = k_new
           cached_v = v_new
        else:
            #Retrieve past histories
            past_k, past_value = kv_cache
            # Linear Algebra Concatenation: Append the new token vector to the timeline axis (dim=1)
            cached_k = torch.cat([past_k, k_new], dim=1)
            cached_v = torch.cat([past_value, v_new], dim=1)

        #update the permanent cache storage tuple reference
        updated_cache = (cached_k, cached_v)

        #3. Compute the speed-p=optimized attention
        #cross-reference the single query vector against all historical keys in the cache
        #q_new [1, 1, 64] * cached_k^T [1, 64, Timeline_Steps]
        scores = torch.matmul(q_new, cached_k.transpose(-2, -1))
        scaled_scores = scores / math.sqrt(self.d_model)
        attention_probs = self.softmax(scaled_scores)

        # Extract context features by multiplying by the complete cached values matrix
        output_context = torch.matmul(attention_probs, cached_v)

        return output_context, updated_cache

#AR-Decoding Simulation pipeline
attn_engine = KVCachedAttention(d_model=64)
cache_store = None

print("="*75)
print(" RUNNING SYSTEM-LEVEL KV-CACHE ATTENTION ENGINE")
print("="*75)

for step in range(1, 4):
    # Simulate a single newly generated input token vector entering the decoder sequence layer
    simulated_new_token = torch.randn(1, 1, 64)
    
    # Process attention pass, feeding in the ongoing cache register
    output, cache_store = attn_engine(simulated_new_token, kv_cache=cache_store)
    
    current_k_cache, current_v_cache = cache_store
    
    print(f"Decoding Step {step}:")
    print(f"  -> Input Vector Shape:        {list(simulated_new_token.shape)} (Only 1 token processed!)")
    print(f"  -> PERSISTENT KEY CACHE SHAPE: {list(current_k_cache.shape)} (Grows linearly over time)")
    print(f"  -> Generated Output Shape:    {list(output.shape)}")
    print("-"*75)

print("KV Cache successfully preserved historical matrices to bypass quadratic waste!")
print("="*75)


