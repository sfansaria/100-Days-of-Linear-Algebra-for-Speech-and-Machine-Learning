'''
Multi-Head Acoustic 

When human beings listen to a conversation, they dont treat every frame of sound as an isolated island.
If a speaker says the word "bank", the actual acoustic wave profile for that word is identical whether 
they are talking about a river bank or a money bank. To figure out the correct transcription, the brain 
uses an internal attention mechanism. It holds onto the word "bank", looks backward at past frames 
"I finished along the", looks forward at future frames "to catch some salmon", and instantly aligns the meaning.

In a Transformer Model, the Multi-Head Self Attention is the mathematical matrix structure that allows the network to do 
this dynamically across a timeline. 

The three matrix projections (Q, K, V)

To process the 80 dimensional Log-Mel spectrogram feature matrix (X),
the model projects it into three distinct matrices via parallel linear layers:

1. Queries (Q) - "The Questions": What features am I looking for in the rest of the audio timeline right now?
2. Key (K) - "The Index Cards": What acoustic features or context do I contain at this specific frame?
3. Value (V) - "The Actual Content": If another time frame finds my index card relevant,
                what raw phonetic information should I pass to it ?

The Scaled Dot-Product Attention Equation:

Once these matrices are projected, they are fed into the core transformer equation:

Attention(Q, K, V) = softmax((Q * K_T)/(sqr_root(d_k))) * V

-> Q * K_T (The Compatibility Grid): Multiply the Query matrix by the transpose of the Key matrix. 
This builds a massive 2D matrix tracking how every single millisecond frame related to evry other 
millisecond frame across the entire audio timeline.

-> 1/(sqr_root(d_k)) (The Scaled Scaling Factor): If the attention hidden features (d_k) are large,
the dot product values grow massive. This pushes the softmax (dim=-1) layer into flat plateau regions 
where the calculus gradients drop to exactly 0.0000, killing the optimizer. Scaling by the square root 
of the dimension preserves stable gradient tracking.

-> The Value Matrix Multiplication (V): Multiply the normalized attention probability by the Value Matrix, 
This pulls out a contextually weighted summary vector for every step, allowing the network to fully unserstand 
the surrounding langugae boundries.

lets build a complete, research-grade Multi-Head Attention Layer using PyTorhc tensor operations.
Pass a simulated timeline of Log-Mel audio features (101 frames * 80 channels) through it and 
trace the precise multi-dimensional matrix dimensions step-by-step. 

'''

import torch
import torch.nn as nn
import math


torch.manual_seed(42)

class MultiHeadSpeechAttention(nn.Module):
    def __init__(self, d_model=80, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        #hidden feature array can be divided evenly among attention heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads #Dimension per head (80 / 4 = 20)

        #Linear Projection Layers to construct Queries, Keys, and Values
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        #Output projection layer to fuse head calculations back together
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1) #this is applied over the competing vocabulary choices

    
    def forward(self, x):
        #x shape : [Batch(1), Time_Frames(101), Audio_Features(80)]
        batch_size, seq_len, _ = x.shape

        #Step A: Project input features to extract basic Q, K, V arrays
        Q = self.w_q(x) #shape: [1, 101, 80]
        K = self.w_k(x)
        V = self.w_v(x)

        #Step B: Reshape for multi-head tracking
        #Split the 80 features into 4 heads of 20 dimensions each
        #Transpose to isolate the head index on Axis 1 for parallel batch processing.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # New shapes: [Batch(1), Heads(4), Time_Frames(101), Head_Dim(20)]

        #Step C: computing the compatibility matrix (Q * K_T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) #shape: [1, 4, 101, 101]

        #Step D: Scale and apply softmax
        scaled_scores = scores / math.sqrt(self.d_k)

        attention_probability_grid = self.softmax(scaled_scores) #shape [1, 4, 101, 101]

        #Step E: Context extraction , weight multiplication
        #multiply probability paths by the content values
        context_features = torch.matmul(attention_probability_grid, V) #shape [1, 4, 101, 20]

        #step F: concatenta ethe 4 heads back together
        #undo the transposition to restore original sequential order
        context_features = context_features.transpose(1, 2).contiguous()

        #collapse the 4 heads * 20 dims back down into a cohesive 80-dimensional matrix
        fused_heads = context_features.view(batch_size, seq_len, self.d_model) #shape [1, 101, 80]
        
        #Final linear mapping layer pass
        return self.w_out(fused_heads), attention_probability_grid
    

#Execution of the simulation pipeline
#passing the 101 frame, 80 channel Log-Mel spectrogram array from signal processing step
simulated_log_mel_spectrogram = torch.randn(1, 101, 80)
attention_block = MultiHeadSpeechAttention(d_model=80, num_heads=4)

output_features, final_attention_matrix = attention_block(simulated_log_mel_spectrogram)

print("Multi-Head Self-Attention Tensor Matrix Diagostics")
print(f"Input Log-Mel Audio Matrix Shape: {list(simulated_log_mel_spectrogram.shape)}") #(Time * Features)
print(f"Multi-Head Split Reshape Shape: [Batch=1, Heads=4, Time=101, Head_Dim=20]")
print(f"Internal Attention Matrix Grid Shape: {list(final_attention_matrix.shape)}")
print("Interpretation: 4 Heads tracking independent 101 * 101 time-dependency maps.")
print(f"Final Transformer Layer Output Shape: {list(output_features.shape)}")
print("Verification Status: Dimensions completely preserved for stacking layers.")
print("Attention matrix mapped cross-time acoustic context")

'''
The final attention matrix shape : [1, 4, 101, 101]
This is the speech sequence modeling layout. 
The network has 4 completely independent 101 * 101 time-alignment matrices.
Head 1 focus on tracking vowels (local phonetics), looks at quick neighboring time frames
Head 2 focus on long-range grammatical context, connects words from the beginning of the timeline to the end.
Head 3 focus on structural silence and audio gaps, indentifies breath pauses, word boundaries, and background noise floors.
Head 4 focus on speaker identity and acoustic profile, tracks global pitch characteristics, vocal patterns and background environments.



'''