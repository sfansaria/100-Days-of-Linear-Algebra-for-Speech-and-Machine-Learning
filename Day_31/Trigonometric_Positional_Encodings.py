'''
In the last code Multi-Head Self-Attention matrix, the first operation involves a giant dot product:
Q*K_T

In matrix multiplication compares all the rows against all the columns simultaneously, 
the self-attention layer has absolutely no concept of time order. If 101 audio frmaes of 
the spectrogram is scrambled into completely random order, the output values inside the 
attention matrix will be exactly the same. But this is fatal flaw in the speech processing.
The sound sequence [/t/ -> /ae/ -> /k/] spells "Tack" but if it reversed the time order, 
it would spell [/k/ -> /ae/ -> /t/] "Cat". The network must know exactly when a frame 
occurs in time.

To rectify the model, a sense of time without using slow, step by step loops, a geometric 
trick is applied. A unique Timestamp vector for every single frame along the timeline is 
generated, and added directly to Log-Mel spectrogram matrix.


Original Log-Mel Feature Matrix X        Trigonometric Positional Matrix PE
       (101 frames × 80 channels)              (101 frames × 80 channels)
      ┌──────────────────────────┐            ┌──────────────────────────┐
    F0│  Raw Acoustic Features   │          F0│  Timestamp at Time 0     │
    F1│  Raw Acoustic Features   │    +     F1│  Timestamp at Time 1     │
      │  ...                     │            │  ...                     │
  F100│  Raw Acoustic Features   │        F100│  Timestamp at Time 100   │
      └──────────────────────────┘            └──────────────────────────┘
                                   │
                                   ▼
             Time-Aware Input Matrix ready for Self-Attention



Instead of of just counting the frames as integers (1, 2, 3, ...), which can
grow massively and cause calculations to explode, use the interlocking 
Sine and Cosine waves at varying frequencies.

-> For the first few channels of the feature vector, use high-frequency waves 
that oscillate rapidly. This lets the model track small, precise micro-seconds 
between neighboring frames (crucial for short consonants like /t/).

-> For the later channels, use the low-frequency waves that stretch out slowly.
This lets the model track long-range macro-time dependencies across the entire 
sentence.

The mathematical equations for a specific time frame position (pos) and a specific
feature channel index (i) inside a model with total feature dimenson (d_model)

PE_pos,2i = sin(pos/(10000 ^ (2i/d_model)))

PE_pos, 2i+1 = cos(pos/(10000 ^ (2i/d_model)))

Even indices (2i): Populated by the Sine wave value
Odd indices (2i+1): Populated by the Cosine wave value

Because of the standard trigonometrical identity rules: 
sin(A+B) = sinA cosB + cosA sin B,
using the matching sine-cosine pairs means that for any two frames,
their relative distance depends purely on the difference between their time steps (delta t), 
regardless of how long the overall audio clip is. The model can instantly figure out if two sounds
happened close together or far apart.


lets generate a structural time grid and add it to the simulated 101 frame audio timeline and 
verify the dimensions.  

'''

import torch
import torch.nn as nn
import math


torch.manual_seed(42)

class SpeechPositionalEncoding(nn.Module):
    def __init__(self, d_model=80, max_len=500):
        super().__init__()
        self.d_model = d_model

        #initialize the baseline grid matrix filled with zeros
        pe = torch.zeros(max_len, d_model)  #max_len = number of time frames, d_model = feature channels

        #construct the time position column vector [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        #compute the scaling factor denominator for the frequency wavelengths
        #step by 2 to process even and odd channels simultaneously
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #Applying the trigonometric wave calculus transformations
        pe[:, 0::2] = torch.sin(position * div_term) #fill the even slots with Sine
        pe[:, 1::2] = torch.cos(position * div_term) #fill the odd slots with Cosine
        

        #add a placeholder batch dimension so it matches neural network shapes: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        #register_buffer ensures this tracking matrix is saved with the model but ignored by optimization gradients
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #x shape: [batch, time_frames, audio features]
        seq_len = x.shape[1]

        #slice the positional matrix to match the exact runtime timeline of the incoming audio
        #then add it element-wise directly to the speech feature matrix
        x_time_matrix = x + self.pe[:, :seq_len, :]

        return x_time_matrix
    

#Execution of the simulation

simulated_speech_matrix = torch.randn(1, 101, 80)
pe_encoder = SpeechPositionalEncoding(d_model=80, max_len=500)

time_aware_speech_matrix = pe_encoder(simulated_speech_matrix)

print("Speech position tracking matrix diagnostics")
print(f"Input Spectrogram Matrix Shape: {list(simulated_speech_matrix.shape)}")
print(f"Internal Positional Matrix Register: {list(pe_encoder.pe.shape)}")
print(f"Time aware processed matrix shape: {list(time_aware_speech_matrix.shape)}")
print("Dimensions are pefectly preserved")
print("Interpretation: Every acoustic frame now carries a unique geometric timestamp")


'''
Now this matrix enters the Multi-Head Self-Attention Layer, the dot product (Q * K_T) wont 
just track raw frequency values, it will calculate cross-time correlations natively,
enabling the model to tell the difference between "Tack" and "Cat"

'''