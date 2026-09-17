'''
Convolutional Subsampling  also called the Audio Front-End  Layer. 
Trigonometric Positional Encoding solve the problem of "Order of time", but they 
do not solve the problem of "Density of time". This introduces the major hardware 
challenge that must be handled before passing the time-stamped speech matrix to
the Multi-head Self-Attention block.

The Quadratic Problem:
In speech processing, audio is processed at incredibly 
high densities. The derived during the speech processing step, cutting 1 second 
of audio into 25 ms windows with 10ms shifts yields 100 time frames per second.

If the user speaks a modest 10 seconds, the input matrix has 1000 time steps.

when this 1000-frame matrix is passed into the Self-Attention layer,
the core calculation is the dot product compatibility grid: Q * K_T

- The Matrix Size: A 1000 step query mutilplied by a 1000 step key results in a
massive 1000 * 1000 attention grid (1 million elements) per attention head.
- The Math scaling: The computational complexity of self-attention grows quadratically (O(T**2)) 
with the timeline length. If the speech clip stretches to 30 seconds, the matrix grid blows 
up to 9 million elements. The GPU will instantly run out of memory trying to track gradients for long audio files.


The Solution:

To prevent the attention layer from choking on dense audio timelines, modern speech transformers (OpenAI's Whisper or standard Conformer Encoder)
inject a 2D Convolutional Subsampling block at the absolute front-end of the model.

Raw Log-Mel Input Matrix           2D Convolution Kernels            Downsampled Compact Matrix
 (1000 frames × 80 channels)             (Stride = 2)                  (250 frames × Hidden_Dim)
 ┌────────────────────────┐              ┌───────────┐                 ┌───────────────────────┐
 │                        │    ➔         │  [3 × 3]  │       ➔         │ Highly dense context  │
 │                        │              └───────────┘                 │ squeezed down by 4x   │
 └────────────────────────┘                                            └───────────────────────┘

This is how linear algebra squeezes the time:
Instead of treating each 10 ms frame as a separate row, 2D Convolutional layers slide 
tiny mathematical filters (kernels) across the spectrogram grid.

- By configuring the convolutional layers with the stride of 2 (filter skips every other step as it rolls),
the algorithm mathematically compresses the timeline.
- Typically two consecutive strided 2D convolutions are chained together. 
Each one cuts in half, resulting in a 4x reduction in the sequence length.


The 1000 frame audio timeline is instantly squashed down to a highly dense, informative 250-frame matrix.
When this compressed matrix hits the Self-Attention layer, the quadratic workload drops from 1 million 
operations to just 62,500 operations (a 93.75% savings in compute and memory footprint)

lets build a standard production-grade Conformer/ Transformer Audio Front-End Subsampling layer.
Pass an 80-channel audio matrix through it and verify the exact dimensional compression down to
the attention-ready hidden feature space.
'''

import torch
import torch.nn as nn

torch.manual_seed(42)

class ConvolutionalSpeechSubsampling(nn.Module):
    def __init__(self, in_channels = 1, out_channels=144, speech_feat_dim=80):
        super().__init__()

        # First 2D Convolution Layer: Cuts time and frequency axes in half
        # Input channels = 1 (Treat  a 2D spectrogram grid like a single-channel grayscale image)
      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

        # Second 2D Convolution Layer: Cuts time and frequency axes in half again
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

        #the resulting frequency feature dimension after two downsamplings
        #80 bins / 2 = 40 bins -> 40 bins / 2 = 20 bins remaining
        #Total flattened feature dimension per time step = 20 bins * 144 channels = 2880

        self.flattened_dim = 20 * out_channels

        #final linear projection to map features down to the attention model's standard hidden size (256 dimensions)
        self.output_projection = nn.Linear(self.flattened_dim, 256)
        self.activation = nn.ReLU()

    
    def forward(self, x):
        #Input x shape: [Batch(1), Time_Frame(100), Mel_Features(80)]
        #PyTorch Conv2D layers expects 4D image tensor: [Batch, Channels, Height, Width]
        #Unsqueeze to add the single-channel axis [Batch, 1, Time_Frame, Mel-Features]
        
        x_4d = x.unsqueeze(1)

        #Run through the strided convolutional blocks 
        x_features = self.activation(self.conv1(x_4d))
        x_features = self.activation(self.conv2(x_features))

        ##check shapes: Batch, Channels ( 144), compressed_time, compressed_features(20)
        batch, channels, comp_time, comp_feats = x_features.shape

        #Permute and flatten the channel and feature dimensions to restore a clean sequential row layout
        # New intermediate shape: [Batch, Compressed_Time, Channels * Compressed_Features]
        x_flat = x_features.permute(0, 2, 1, 3).contiguous().view(batch, comp_time, -1)
        
        #Project down to standard Transformer attention input dimension size
        final_attention_ready_matrix = self.output_projection(x_flat)

        return final_attention_ready_matrix



#Simulating a dense 10second audio clip, clip with 1000 Frames * 80 Log-Mel channels
dense_speech_spectrogram = torch.randn(1, 1000, 80)
subsampling_block = ConvolutionalSpeechSubsampling()

compressed_attention_input = subsampling_block(dense_speech_spectrogram)

print("Convolutional Front-End Subsampling Diagnostics")
print(f"Raw Audio Spectrogram Input Shape: {list(dense_speech_spectrogram.shape)} (1000 Time Steps)")
print(f"Hardcoded Feature Extraction Dimension: 80 Log-Mel channels")
print(f"Subsampled compressed output shape: {list(compressed_attention_input.shape)}")
print("Time Dimension Compression: 1000 steps squeezed to 250 steps (4x Reduction)")
print("Feature Projection Vector Size: Hidden attention feature dim set to 256")
print("Audio compressed to prevent the quadratic attention explosion.")


'''
When this [1, 250, 256] matrix passed through the Multi-Head Self-Attention Layer in the first step,
the system can easily calculate attention alignments over long paragraphs of spoken text without over-allocating GPU memory.
'''