'''
Continuous Audio Waveforms are chaotic and complex, making them difficult
to analyse and process as they are infinitely variable. However, Text models like 
GPT excel because they process discrete, distinct tokens (word/characters).
To make speech generation  efficient, transform the continuous audio spectrograms 
into discrete ""acoustic words" or tokens. This process is called Vector Quatization (VQ).

When listening to a voice of person speaking, instead of saving the every single microscopic
decimal point of the their Mel-spectrogram, you can save  it in a massive internal dictionary called a codebook.

-> The codebook is a matrix containing a fixed number of vectors (example: 1024 vectors), where 
each row represents a "prototype" acoustic sound shape (like a clean vowel, a sharp breatthe, or a nasal sound).

-> For every incoming audio frame, the model performs a geometric search across the dictionary.
It finds the closest matching/ closest to the current audio frame vector, finds it quickly and 
outputs that dictionary index 

Continuous Audio Feature Frame       Geometric Search              The Codebook Matrix
      [ 0.72, -1.2, 0.44... ]    ──►  Find Closest Row  ──►  Row 412: [ 0.70, -1.1, 0.40... ]
                                                                       ▼
                                                             Output Discrete Token: 412


By doing this, a continuous 1-second vocal recording is compressed into a clean list of whole numbers:
[412, 12, 994, 87]. A standard language transformer can now predict speech just like it predicts words in a sentence.

The mathematical process: Using the calculus

Finding the closest dictionary entry is a hard threshold operation (argmin), the function 
is mathematically non-differentiable. If derivative is calculated using this function, 
the gradient collapses to zero. This would break the backpropagation process.

To solve this, researchers uses a technique called the Straight-Through Estimator (STE).
During the forward pass, the model finds quickly the discrete codebook vector.  
But during the backward pass, the algorithm completely copies the gradients from the output 
and passes them straight through the threshold unchanged to the previous layer, allowing the model
to optimize smoothly.

'''

import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings = 1024, embdedding_dim = 80):
        super().__init__()
        self.embedding_dim = embdedding_dim
        self.num_embeddings = num_embeddings

        #The learned Acoustic Codebook Dictionary Matrix [shape: (num_embeddings, embedding_dim)]
        self.codebook = nn.Embedding(num_embeddings, embdedding_dim)
        
        #Uniform initialization of the dictionary vectors
        self.codebook.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)

    def forward(self, x):
        #Input x shape: [Batch(1), Time(32), Features(80)]

        batch, time, feats = x.shape

        #Flatten the input to a 2D matrix for distance calculations
        
        flat_x = x.view(-1, self.embedding_dim)

        #Geometric Euclidean Distance between every frame and every codebook vector
        #Formula:  ||a - b||^2 = a^2 + b^2 -2ab

        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2*torch.matmul(flat_x, self.codebook.weight.t()))
        
        #discrete token extraction: find the index of the closest dictionary row
        encoding_indices = torch.argmin(distances, dim=1) #shape : [32]

        #quantize: retrieve the matching vectors from the codebook using the indices
        quantized_vectors = self.codebook(encoding_indices).view(batch, time, feats)

        #straight-through estimator (STE) calculus hack:
        # quantized_vectors has no gradient tracking. x has gradient tracking.
        # By writing it like this, quantized_vectors_ste gets the values of quantized_vectors,
        # but copy-pastes the calculus backward pass directly from x
        
        quantized_vectors_ste = x + (quantized_vectors - x).detach()

        #VQ Loss Formulation:
        # small secondary loss to force the encoder to output vectors 
        # that stay close to the codebook dictionary entries.
        commitment_loss = torch.mean((quantized_vectors.detach() - x)**2)

        return quantized_vectors_ste, encoding_indices.view(batch, time), commitment_loss

#Simulated 32-frame matrix of acoustic voice features
simulated_audio_frames = torch.randn(1, 32, 80, requires_grad=True)
vq_bottleneck = VectorQuantizer(num_embeddings=1024, embdedding_dim=80)
optimizer = optim.AdamW(vq_bottleneck.parameters(), lr = 0.01)

print("Vector Quantization Process: ")
optimizer.zero_grad()

quantized_out, discrete_tokens, vq_loss = vq_bottleneck(simulated_audio_frames) 

reconstruction_loss = torch.mean((quantized_out - torch.randn_like(quantized_out))**2)
total_loss = reconstruction_loss + 0.25 * vq_loss

total_loss.backward()
optimizer.step()

print(f"Continuous Input Tensor Shape: {list(simulated_audio_frames.shape)}")
print(f"Quantized Discrete Token Array: {discrete_tokens.squeeze(0).tolist()[:10]}..(Truncated)")
print(f"Continuous Audio is mapped to integer indices")
print(f"Quantized Output Tensor Shape: {list(quantized_out.shape)}")
print(f"Straight-Through Gradient Verification: {simulated_audio_frames.grad is not None}")
print("Gradient bypassed the argmin threshold safely")
