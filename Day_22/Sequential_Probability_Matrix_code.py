'''
lets demonstrate how an Automatic Speech Recognition (ASR) system processes an ongoing timeline of sound. 
We take an audio matrix (100 time frames × 80 acoustic features), pass it through a neural network, generate a 
Phoneme Probability Sequence Matrix, and apply Sequential Negative Log-Likelihood MLE to optimize the weights.
'''


import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for tracking consistency
torch.manual_seed(42)

# --- STEP 1: Setup Dimensions for the Sequence Grid ---
num_time_frames = 100       # 1 second of audio cut into 100 slices (time steps)
num_mel_features = 80       # 80 Mel-filterbank frequency channels per frame
num_phoneme_classes = 4     # Simple dictionary: [0: /blank/, 1: /k/, 2: /æ/, 3: /t/]

# Initialize input audio sequence. Shape: [Batch Size, Time Steps, Audio Features]
simulated_audio_sequence = torch.randn(1, num_time_frames, num_mel_features)


# --- STEP 2: Build a Sequential Acoustic Network ---
class SpeechSequenceModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        # Linear layer maps the 80 acoustic features to our 4 phoneme class slots
        self.acoustic_encoder = nn.Linear(in_features, num_classes)
        # Softmax applied to the final dimension to create valid probability fields
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x input shape: [1, 100, 80]
        # We strip out the batch dimension to feed a pure 2D sequence matrix [100, 80] to the linear layer
        x_2d = x.squeeze(0) # New shape: [100, 80]
        
        logits = self.acoustic_encoder(x_2d) # Output shape: [100, 4]
        probabilities = self.softmax(logits) # Final shape: [100, 4]
        return probabilities

model = SpeechSequenceModel(num_mel_features, num_phoneme_classes)
optimizer = optim.AdamW(model.parameters(), lr=0.05)


# --- STEP 3: Define the Ground-Truth Sequential Targets ---
true_alignments = torch.zeros(num_time_frames, dtype=torch.long)
true_alignments[0:31] = 1   # /k/
true_alignments[31:71] = 2  # /æ/
true_alignments[71:100] = 3 # /t/


print("="*75)
print(" RUNNING SEQUENTIAL MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
print("="*75)

# --- STEP 4: The Sequential Training Loop ---
for epoch in range(3):
    optimizer.zero_grad()
    
    # Forward Pass: Generate the Probability Sequence Matrix
    # The model now natively returns a perfect [100, 4] matrix
    prob_sequence_matrix = model(simulated_audio_sequence) 
    
    # --- STEP 5: Calculate Sequential Negative Log-Likelihood (MLE) ---
    total_sequence_nll = 0.0
    eps = 1e-12 
    
    for t in range(num_time_frames):
        correct_phoneme_idx = true_alignments[t].item()
        
        # Extract the specific probability the model gave to the true sound at this millisecond
        prob_of_target = prob_sequence_matrix[t, correct_phoneme_idx]
        
        # Accumulate the Negative Log-Likelihood across time
        total_sequence_nll -= torch.log(prob_of_target + eps)
        
    # Average the loss across the 100 time frames
    average_sequence_loss = total_sequence_nll / num_time_frames
    
    # Execute matrix calculus and update the acoustic layer weights
    average_sequence_loss.backward()
    optimizer.step()
    
    print(f"Iteration {epoch+1}:")
    print(f"  -> Total Sequence NLL Error: {total_sequence_nll.item():.2f}")
    print(f"  -> Average Frame Loss:       {average_sequence_loss.item():.4f}")
    
    # Inspect a snapshot of frame 50 (which is the phoneme /æ/)
    frame_50_probs = prob_sequence_matrix[50].tolist()
    print(f"  -> Frame 50 Probabilities:   [/blank/: {frame_50_probs[0]:.3f}, /k/: {frame_50_probs[1]:.3f}, /æ/: {frame_50_probs[2]:.3f}, /t/: {frame_50_probs[3]:.3f}]")
    print("-"*75)

print("The sequential probability matrix shape is corrected and optimized!")
