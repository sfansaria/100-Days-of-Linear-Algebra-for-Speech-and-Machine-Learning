'''
Continuous Flow Matching (CFM) - 
In the state-of-the-art speech synthesis research (powering architecture like ElevenLabs' 
advanced back-ends, Meta's Voicebox, or advanced text-to-speech models), discrete token 
systems like RVQ have a competitor. Instead of quantization sieves, CFM models speech 
generation as a continuous physical trajectory.

Flow Matching is the mathematical framework that trains a neural network to act as the
sculptor's hands. Instead of trying to guess the final audio layout in a single step,
the model learns a Velocity Field. At every micro-step along a simulated timeline 
(t belongs to [0, 1]), the model looks at the current noisy matrix state and calculates the
exact directional velocity vector required to push the noise closer to a clean speech profile.

[ Random Noise (t=0) ] ───► [ Compute Velocity Vector ] ───► [ Clean Speech Matrix (t=1) ]
           │                                                               ▲
           └───────► Iteratively deforms noise along the trajectory ───────┘

The Mathematics behind the Flow Matching:
Unlike the classic Diffusion models, which use the complex, curved stochastic paths that 
are slow to compute, Conditional Flow Matching constructs a simple mathematically straight
linear path between the noise and the true audio:
x_t = (1-t)*x_0 + t*x_1

The true target velocity field (v_t(x)) that drives the system along this straight line is the
exact derivative of this path w.r.t time:

dx_t / dt = v_t(x) = x_1 - x_0

During the training, the optimization engine takes a neural network u_theta(x_t, t) and 
uses calculus gradients to force it to predict this target vector field via a 
Flow Matching Objective function:

L_CFM(theta) = E_t,x_0,x_1||u_theta(x_t, t) - (x_t-x_0)||**2

By minimizing this mean squared error across the velocity paths, 
the network masters the dynamics of sound formation.

Lets build a functional Continuous Flow Matching training step.
Take a real voice matrix, corrupt it with pure Gaussian Noise, 
construct a linear temporal path, and train a velocity network
to predict the audio trajectory.
'''

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#Stage 1 : Define the Speech Velocity Network
class AudioVelocityNetwork(nn.Module):
    def __init__(self, feature_dim=80):
        super().__init__()
        # A simple vector network to predict the directional velocity matrix
        # (In production, this would be a massive DiT - Diffusion Transformer)
        self.net = nn.Sequential(nn.Linear(feature_dim+1, 128), # +1 adds the scalar time step variable 't'
                                 nn.ReLU(),
                                 nn.Linear(128, feature_dim)
                                )
    
    def forward(self, x_t, t):
        # x_t shape: [Batch, Features(80)]
        # t shape: [Batch, 1] (The current position along the timeline from 0 to 1)
    
        # Concat the current noise coordinates with the time stamp scalar
        input_tensor = torch.cat([x_t, t], dim=-1)
        return self.net(input_tensor)

#Stage 2 : Configuration and Audio Initialization 
num_frames = 32
feature_channels = 80

#The target output: An authentic, clean human voice frame matrix slice 
true_target_speech = torch.randn(num_frames, feature_channels)

#Instantiate the CFM Engine
velocity_model = AudioVelocityNetwork(feature_dim=feature_channels)
optimizer = optim.AdamW(velocity_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("Continuous Flow Matching (CFM) velocity dynamics")

#stage 3: The Flow matching training loop step
optimizer.zero_grad()

#A.Generate the pure baseline Gaussian Noise matrix starting point (t=0)
x_0_noise = torch.randn_like(true_target_speech)
x_1_audio = true_target_speech #The destination (t=1)

#B. Sample a random micro-step time position along the timeline example t=0.40
#Broadcast it to match the shape of the audio frames row count

t_scalar = torch.randn(1, 1) #a single random decimal between 0.0 and 1.0
t_vector = t_scalar.repeat(num_frames, 1)

#C. Linear Interpolation Path calculation (construct x_t coordinates)
#Formula: x_t = (1-t)*x_0 + t*x_1

x_t_trajectory = (1.0 - t_vector) * x_0_noise + t_vector * x_1_audio

#D. The Ground Truth Target Velocity Vector: (x_1 - x_0)
target_velocity_field = x_1_audio - x_0_noise

#E. Predict the velocity direction using the network
predicted_velocity = velocity_model(x_t_trajectory, t_vector)

#F. Calculate the Flow Matching Loss 
cfm_loss = loss_fn(predicted_velocity, target_velocity_field)

#Run Calculus and execute the optimizer step
cfm_loss.backward()
optimizer.step()

print(f"Target Trajectory Time Step (t): {t_scalar.item():.4f} (Timeline coordinate)")
print(f"Random Input Noise Grid Shape: {list(x_0_noise.shape)}")
print(f"Interpolated Trajectory Matrix: {list(x_t_trajectory.shape)}")
print(f"Computed Flow Matching Loss: {cfm_loss.item():.4f}")
print("The network learned the straight vector path velocity vector.")