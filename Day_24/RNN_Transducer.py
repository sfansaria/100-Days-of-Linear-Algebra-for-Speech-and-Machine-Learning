'''
CTC is a brilliant step forward, but it has one limitation, it assumes that the model's prediction at each time step is 
completely independent of the what it predicted some milliseconds ago. But in reality, the model's prediction at time t is
highly dependent on what it predicted at t-1. . If the model predicted a "b" at t-1, then it is more likely to predict "c" at t 
than predicting "a". 
This problem introduces the RNN Transducer, which fixes the model by setting up two separate neural networks,
that feed  their calculations into a joint network.
- The first model is the Acoustic model (Transcription Network) which acts as a microphone processor. 
It looks frame by frame at the incoming audio features (Mel-spectogram) to guess what sound is being spoken right now.
- The second model is the Predictor model (Language Network) which acts as an internal text autoregressive engine.
It completely ignores the audio and simply reads the history of the characters that the system have already predicted. 
It tries continuously to guess what the next character or token is most likely based on pure grammar and spelling rules.


The Joint Network Matrix - The Two models meet inside a joint network. Instead of outputting a simple 2D sequence
over time (like a CTC model), the RNN Tranducer cross-references them to build a massive 3D Probability Lattice Grid.
Grid Coordinates = (Time steps T, Label steps U, Phoneme Classes)

As the audio streams in, the model tracks its progress through the lattice. 
Moving the horizontally means a time step has passed processing the sound,
moving vertically means a text character has been emitted. 
This structural design is what allows modern phone assistants like Siri or Google Assistant or Alexa
to type out words on the screen instantly as they being spoken, without waiting for the audio file to finish.

lets simulate using PyTorch to map the Joint Network. This code shows exactly how the separate Audio features and text history
vectors are combined via broadcasting to generate the signature 3D RNN-T probability lattice grid.

'''

import torch
import torch.nn as nn

torch.manual_seed(42)

#sequence dimensions
batch_size = 2
T_time_frames = 40 #40 audio frames from the microphone input
U_text_history = 5 #system decodes 5 text characters at a time
num_phoneme_classes = 4 #[0: blank, 1: 'c', 2:'a', 3: 't']

#simulate the separate audio and text embeddings
# #In real pipeline, these would be the hidden outputs from the RNN or Conformer layers
acoustic_hidden_features = torch.randn(batch_size, T_time_frames, 64)   #64 hidden features from the acoustic model
predictor_hidden_features = torch.randn(batch_size, U_text_history, 64) #64 hidden features from the predictor model

class RNN_Joint_Network(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()

        #Two independent projection vectors to align dimensions
        self.project_audio = nn.Linear(hidden_dim, hidden_dim)
        self.project_text = nn.Linear(hidden_dim, hidden_dim)

        #Final classification head mapping back to vocabulary classes
        self.classify_head = nn.Linear(hidden_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1) #applied over the vocabulary axis
    
    def forward(self, h_audio, h_text):
        #project the audio and text features to the same hidden dimension
        phi_audio = self.project_audio(h_audio) #shape: (batch_size, T_time_frames, hidden_dim)
        phi_text = self.project_text(h_text) #shape: (batch_size, U_text_history, hidden_dim)
        
        # To let every single time frame cross-reference with every single past text choice,
        # expand their matrix dimensions so they add together natively:
        # Expand audio to: [Batch(1), Time(40), 1,       Dim(64)]
        # Expand text to:  [Batch(1), 1,        Text(5), Dim(64)]

        joint_features = torch.tanh(phi_audio.unsqueeze(2) + phi_text.unsqueeze(1))

        #map the combined 4D tensor down to the vocabulary class distribution
        logits = self.classify_head(joint_features) ## Shape: [1, 40, 5, 4]
        
        return self.log_softmax(logits)
    


#Initialize the RNN Transducer Joint Network
joint_net = RNN_Joint_Network(hidden_dim=64, num_classes=num_phoneme_classes)

#execute the forward pass to generate the 3D probability lattice grid
rnnt_probability_grid = joint_net(acoustic_hidden_features, predictor_hidden_features)

print("RNN-Transducer Lattice Generation")
print(f"Audio Encoder Core State Vector: {list(acoustic_hidden_features.shape)}")
print(f"Text Predictor Core State Vector: {list(predictor_hidden_features.shape)}")
print(f"RNN-T Probability Lattice Grid: {list(rnnt_probability_grid.shape)}")
print("Dimensions Breakdown: [Batch, Time_Frames (T), Text_History (U), Vocab_Classes]")
print("Target Optimization Check:")

sample_coordinate_probs = torch.exp(rnnt_probability_grid)

print(f"Probability field at Time Frame 12, Text Token 3:")
print(f"{sample_coordinate_probs.tolist()}")
print("Broadcasting successfully fused the acoustic and language state fields.")