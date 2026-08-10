'''
In Speech ml and automatic speech recognition (ASR) perpective, audio is highly variable, dynamic and noisy.
While standard vision models, process clean, static pixel grids, speech models map moving time-series acoustic signals into predictions.

The Jacobian and Hessian matrices are crucial for handling the non-linear, distorted paths of human speech.
They transition from mathematical abstracts to fundamental tools for speech processing in several processing in several specific ways.

The Jacobian Matrix - If the gradient is a vector containing the first partial derivatives of a single function, 
Then Jacobian is a full matrix containing all the first partial derivatives for multiple equations simultaneously. 
This is what passes gradients across entire neural network layers during backpropagation. 
The Jacobian measures how a multi-values output vector changes relative to a multi-valued input vector.
In speech, this represents the relationship between an incoming acoustic feature frame and a model's predicted phonetic outputs.
In real-world speech application of Jacobian is in Feature enhancement and Vector Taylor Series (VTS). A core challenge in ASR 
is environment mismatch (for example: trying to understand an engine-noised voice when the model was trained on a silent recording). 
Historically and conceptually, Vector Taylor Series (VTS) adaptation fixes this.

The relationship between clean speech(x), additive noise(n) and channel distortion (h) to form noisy speech (y) is highly non-linear 
in the log-spectral domain:
                            y ~ x + h + log(1 + exp(n-x-h))

To decode the audio, the model must linearise this equation around a guess. It uses a Jacobian matrix(J) to track 
how sensitive the noisy speech features (y) are to changes in the clean speech features(x).


If you have an acoustic model that takes N input features and outputs M phonetic predictions, 
the Python matrix elements are structured precisely like this:

# Assuming M = 3 phoneme classes, N = 4 audio features
jacobian_matrix = [
    [ ∂y₀/∂x₀,  ∂y₀/∂x₁,  ∂y₀/∂x₂,  ∂y₀/∂x₃ ],  # Row 0: Phoneme 0 sensitivity
    [ ∂y₁/∂x₀,  ∂y₁/∂x₁,  ∂y₁/∂x₂,  ∂y₁/∂x₃ ],  # Row 1: Phoneme 1 sensitivity
    [ ∂y₂/∂x₀,  ∂y₂/∂x₁,  ∂y₂/∂x₂,  ∂y₂/∂x₃ ]   # Row 2: Phoneme 2 sensitivity
]


Each row maps a specific output frequency bin (yi) and 
each column maps an input log-mel filterbank feature (xj).

The Jacobian calculates the "distortion path", allowing a noise-reduction system to strip away 
background humming dynamically without corrupting the speaker's actual voice profile.


The Hessian Matrix - This matrix maps out the second-order partial derivatives. It measures the curvature of your (landscape) loss function surface. 
In speech training, the loss lanscape is uniquely challenging because human language has vast time variations 
(example - speaking a word quickly vs drawing a out a vowel)
It helps adaptive optimizers determine if a slope is flattening out or dropping off steeply.

  First-Order (Gradient/Jacobian)       Second-Order (Hessian)
       "Which direction is down?"          "How sharply is the path bending?"
                \                                      \__
                 \                                        \___
                  ▼                                           ▼

                  
Modern speech networks use architectures like Connectionist Temporal Classification (CTC) or RNN-Transducers to map audio frames to characters.
Because text and audio sequences do not line up perfectly, the loss landscape is eceptionally rocky, full of sharp ravines and flat plateaus.

Saddle points vs Real Drops - Standard optimization methods like Gradient Descent easily get stuck in "saddle points"
(areas that look flat to a first-order derivative but bend up or down elsewhere) . 
The Hessian matrix checks the eigenvalues of the second derivatives. If the eignenvalues are mixed positive and negative, 
the model knows it is on a saddle point and bypasses it.

Second-Order Optimization (Natural Gradient): Speech training benefits significantly from K-FAC (Kronecker-factored Approximate Curvature), 
which uses an approximation of the Hessian. If the speaker has a rare accent, a standard gradient step might completely erase the model's 
progress on general English . The Hessian corrects the step size, by slowing the updates down in hyper-sensitive error curves and 
speeding them up over flat terrains. 
'''
#Lets simulate a real scenario: mapping Acoustic Audio Frames (Log-Mel Filterbanks) to a Phonetic Class Probabilities using a neural network layer.

import torch 
import torch.nn as nn


torch.manual_seed(42)   

#Step1 : Defining a speech acoustic  mapping model
class SpeechLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        return self.activation(self.linear(x))

#Dimensions setup 
#Input: 13-dimensional MFCCs or Log-Mel features for a single frame
#Output: 5 phonetic classes (example: /ah/, /ee/, /s/, /t/, /m/)
num_acoustic_features = 13
num_phoneme_classes = 5
model = SpeechLayer(num_acoustic_features, num_phoneme_classes)


#simulate 1 incoming audio frame
audio_frame = torch.randn(num_acoustic_features, requires_grad=True)

#Step 2: Calculate the Jacobian  Mtarix
#The Jacobian tells us: How changing each individual acoustic features 
#impact the probability of each individual phoneme class predictions

#PyTorch functional API expects a function wrapper that takes the input tensor
def model_forward_wrapper(x):
    return model(x)

#Compute the Jacobian matrix 
#Dimensions will be [num_phoneme_classes, num_acoustic_features] (5 * 13)
jacobian_matrix = torch.autograd.functional.jacobian(model_forward_wrapper, audio_frame)

#Step 3: Calculate the Hessian matrix 
#The Hessian tells us about the curvature of the loss function landscape.
#define a simple speech loss function (binary cross entropy, mean squared error )

 #target phoneme alignment 
target_phoneme = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
loss_fn = nn.MSELoss()

# The Hessian maps how the gradient of the loss changes with respect to the model weights
# ,extract the weight parameter tensor from our model layer
weight_tensor = model.linear.weight # Dimensions: [5, 13] (65 total parameters)

def loss_wrapped_for_weights(w_param):
    original_weight = model.linear.weight

    model.linear.weight = nn.Parameter(w_param)

    #Forward_pass
    predictions = model(audio_frame)
    loss = loss_fn(predictions, target_phoneme)
    
    #Reset the original weights
    model.linear.weight = original_weight
    return loss 

# Compute the Hessian matrix
# Dimensions will be [65, 65] tracking interaction between every parameter pair
hessian_matrix = torch.autograd.functional.hessian(loss_wrapped_for_weights, weight_tensor)

print(" SPEECH ML CALCULUS MATRIX DIAGNOSTICS")
print(f"Audio Frame Shape: {audio_frame.shape}")
print(f"Phoneme Class Predictions Shape: {model(audio_frame).shape}")
print(f"Jacobian Matrix shape: {jacobian_matrix.shape}")
print("Interpretation : Rows = Phonemes, Columns = Acoustic  Features")
print(f"Example value J[0,0]:         {jacobian_matrix[0, 0].item():.4f}")
print("(Sensitivity of Phoneme 1 relative to Audio Feature 1)")
print(f"HESSIAN MATRIX SHAPE:           {hessian_matrix.shape}")
print(f"Flattened parameter count:    {weight_tensor.numel()} parameters")
print(f"Resulting Hessian footprint:  {hessian_matrix.view(65, 65).shape}")
print("Interpretation: Maps the mathematical curvature of the speech error landscape.")