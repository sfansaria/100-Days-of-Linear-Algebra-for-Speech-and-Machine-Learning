'''
After finishing the Calculus part starting from 
Partial Derivatives -> Gradients -> Chain Rule -> Jacobian and Hessian Matrix comes the 
Optimization of the Algorithms. This step answers the question "Now that i know which way is down,
how do i actually walk down the mountain to find the lowest error.

let us imagine skiing down a foggy mountain situation at the top of the mountain and 
the thick fog rolls in. You cannot see the lodge at the bottom (the perfect analogy  
to speech model configuration). You can only feel the slope of the snow beneath of your skis.

In speech ml, this mountain represents the loss landscape:
-> The coordinates: It is the current position on the mountain is determined by millions of neural network weights.
-> The steepness (Gradient/Slope): If you tilt forward, you feel how steep the hill is in a specific direction.
-> The Fog(Acoustic Noise): Speech audio contains background noise, accents and varying speeds. This makes the 
snow surface high uneven, filled with random icy patches, small bumps, and sudden drops.

An Optimization Algorithm is the set of rules the brain uses to navigate down this mountain safely and quickly.

Method 1 - Stochastic Gradient Descent (SGD) - 
Intuition - You feel the slope at your current spot, push off and take a fixed-size
step directly downward. Then you stop, feel the slope again and take another step.  

The speech problem - If your step size (learning rate) is too big, you will fly off a cliff.
If it is too small, you will stuck in a tiny dip in the snow (local minimum) and think you 
reached the bottom even though the lodge us miles away.

Method 2 - Momentum
Intuition - Instead of stopping after every step, you let gravity build your speed.
If you are sliding down a long, steady slope, you accumulate momentum to glide right
over small bumps and dips in the snow.

The speech benefit - Human speech patterns have a lot of repetitive structures. Momentum
helps the model slide past random noise variations in individual audio frames.

Method 3 - Adam (Adaptive Moment Estimation)
Intution - This is like having high-tech, adaptive skis. If the snow becomes incredibly
steep and rocky, your skis automatically dig in and slow you down to prevent a crash. If you hit a long,
flat plateau, your skis immediately accelerate to get you across it quickly.

The speech benefit - Some words or phonemes appear constantly (like "the" or "and"), while rare words appear 
infrequently. Adam dynamically scales the step size for each parameter so rare acoustic features still get 
learned without destroying the general weights. 
'''

#Lets train a Speech Classifier. It takes a speech features, calculates the gradient using calculus, and 
#uses the Adam Optimizer to navigate the loss landscape.

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#Simulate a speech data 
#Imagine 100 audio samples. Each sample has 13 Mel-Frequency Cepstral Coefficients (MFCCs)
# The target is binary: 0 = "Silence", 1 = "Active Speech"

X_speech_features = torch.randn(100, 13)
Y_speech_labels = torch.randint(0, 2, (100, 1)).float()

#Define a simple speech network
class VoiceActivityDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(13, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.layer(x))

model = VoiceActivityDetector()

#The error metric #the mountain surface
loss_fn = nn.BCELoss()

#The optimizer (How to navigate the mountain)
learning_rate = 0.01 #size of the initial step
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starting the Optimization trajectory")
#The optimization loop
#Look at the data 5 times (epochs), taking steps down the hill each time
for epoch in range(5):
    predictions = model(X_speech_features) #1. Forward Pass (skiing to a new spot on the hill)
    loss = loss_fn(predictions, Y_speech_labels) #2. Calculating the elevation/loss (checking the altitude)
    optimizer.zero_grad() #3. Clear old calculation history
    loss.backward() #4. Backward Pass (Calculus determines the local slope)
    optimizer.step() #5. The Optimizer Step (Moving down the slope based on Adam's rule)

    print(f"Epoch {epoch+1}: Current Loss Elevation = {loss.item():.4f}")
