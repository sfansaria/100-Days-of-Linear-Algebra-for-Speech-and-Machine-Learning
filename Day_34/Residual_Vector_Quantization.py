'''
Residual Vector Quantization (RVQ)-
If only one codebook is used to compress the voice, 
the model is forced to make a compromise between the different parts of the signal.
The single codebook will use almost all of its slots to memorize the core vowel and 
consonants (text-like data), because those have the highest mathematical energy.
The fine acoustic details like a speaker's unique vocal texture, subtle emotional inflections,
background breathing, or the crisp airiness of a whisper are completely lost. 
The reconstructed voice sounds flat, robotic, or muffled.
To capture these tiny details without exploding the codebook size to an unmanageable millions 
of rows, researchers introduced the Residual Vector Quantization (RVQ). This architecture
is the core mathematical engine behind industry-defining generative audio models like 
Meta's EnCodec, Google's Soundstream, and open-source models like Descript's AudioCraft.


Continuous Hidden Speech Input (x)
                 │
                 ▼
     ┌───────────────────────┐
     │ 1. Codebook Stage 1   │ ──► Snaps to Coarse Sound Shape (example - Token 412)
     └───────────┬───────────┘
                 ▼ (Calculates the leftover error: x - Quantized_1)
     ┌───────────────────────┐
     │ 2. Codebook Stage 2   │ ──► Snaps to Speaker Vocal Texture (e.g., Token 88)
     └───────────┬───────────┘
                 ▼ (Calculates the next remaining error)
     ┌───────────────────────┐
     │ 3. Codebook Stage 3   │ ──► Snaps to Fine Whispers / Breath Details (e.g., Token 5)
     └───────────────────────┘

Stage 1 - extracts the absolute loudest, coarse phonetic layout
Stage 2 - takes the error left behind by Stage 1 and snaps it to the secondary dictionary,
capturing the speaker's vocal tone
Stage 3 - takes the leftover micro-error from Stage 2 and snaps it to a third dictionary,
capturing the silent background breaths, and acoustic air.

By multiplying these choices together sequentially, a 1-second vocal slice is now represented by a 
small vertical grid of tokens: [412, 88, 5]. This multi-layered token representation allows 
the generative models to clone and synthesize speech with extreme, lifelike fidelity. 
'''

import torch
import torch.nn as nn

torch.manual_seed(42)

class SingleQuantizerStage(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=80):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, x):
        #Flatten the input to a 2D matrix for geometric distance tracing: [Time*Batch, 80]
        flat_x = x.view(-1, self.embedding_dim)

        #Matrix distance expansion: ||a-b||^2 = a^2 + b^2 -2ab
        distance = (torch.sum(flat_x**2, dim=1, keepdim=True)
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.codebook.weight.t()))
        
        #Find the index of the closest dictionary row
        encoding_indices = torch.argmin(distance, dim=1)
        quantized_vectors = self.codebook(encoding_indices).view(x.shape)

        #Straight-Through Estimator calculus gradient copy-paste bypass
        quantized_vectors_ste = x + (quantized_vectors - x).detach()
        return quantized_vectors_ste, encoding_indices

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_stages=3, num_embeddings=512, embedding_dim=80):
        super().__init__()
        self.stages = nn.ModuleList([SingleQuantizerStage(num_embeddings, embedding_dim) for _ in range(num_stages)])
    
    def forward(self, x):
        #x shape: [Batch(1), Time(32), Features(80)]
        residual = x
        total_quantized_output = torch.zeros_like(x)

        all_stage_indices = []
        all_stage_losses = []
        
        # Loop recursively through our codebook sieve stack
        for stage_idx, quantizer_stage in enumerate(self.stages):
             # 1. Snap the current residual error to the closest row in this stage's dictionary
            quantizer_step, indices = quantizer_stage(residual)

            # 2. Accumulate the quantized vectors to build the final high-fidelity audio shape
            total_quantized_output = total_quantized_output + quantizer_step

            # 3. Track the quantization error (residual) for the NEXT stage to look at
            # Residual = True input at this step minus what this stage successfully captured
            residual = residual - quantizer_step

            # Save the discrete token coordinates for generative language modeling usage
            all_stage_indices.append(indices)

            #Track the remaining error metric for this layer
            stage_variance = torch.mean(residual**2)
            all_stage_losses.append(stage_variance)
        
        # Combine discrete tokens into a clean grid matrix layout: [Num_Stages, Time_Steps]
        stacked_tokens = torch.stack(all_stage_indices, dim=0)

        return total_quantized_output, stacked_tokens, all_stage_losses

#Residual Simulation Pipeline
# Simulated 32-frame continuous audio spectrogram feature matrix
continuous_voice_features = torch.randn(1, 32, 80)
rvq_pipeline = ResidualVectorQuantizer(num_stages=3, num_embeddings=512, embedding_dim=80) 

quantized_out, discrete_token_grid, stage_errors = rvq_pipeline(continuous_voice_features)


print("="*75)
print(" RESIDUAL VECTOR QUANTIZATION (RVQ)")
print("="*75)
print(f"1. Continuous Input Vector Shape:    {list(continuous_voice_features.shape)}")
print(f"2. FINAL RVQ TOKEN GRID SHAPE:       {list(discrete_token_grid.shape)}")
print("   -> Interpretation: Columns = Time steps, Rows = Hierarchical Codebook Stages.")
print("-"*75)
print("3. ACCURACY REFINEMENT TRACKING TIME SEGMENTS (Lower error = More nuance caught):")
print(f"   -> Stage 1 Residual Error (Coarse): {stage_errors[0].item():.4f}")
print(f"   -> Stage 2 Residual Error (Tone):   {stage_errors[1].item():.4f}")
print(f"   -> Stage 3 Residual Error (Breath): {stage_errors[2].item():.4f}")
print("-"*75)
print("4. PRODUCTION GENERATIVE LOOK:")
print(f"   -> First 4 audio time frames mapped down to discrete codebook grids:")
print(discrete_token_grid[:, :4].tolist())
print("="*75)
print("Chained codebooks recursively extracted fine vocal variance traits.")
print("="*75)