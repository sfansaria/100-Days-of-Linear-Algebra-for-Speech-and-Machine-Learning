'''
Lets simulate a Multi-Task Speech Foundation Layer. This shared neural network layer is responsible for 
processing acoustic audio features to solve two tasks simultaneously. 
Task 1 -> Transcription Logits and Task 2 -> Translation Logits

The code calculates the independent gradients for each task, 
checks for mathematical conflict using the vector dot product, and performs vector surgery 
to strip away destructive interference before updating the shared weights.
'''

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#A Shared Multi-Task Speech LAYER
in_features = 13 #13 Acoustic features (Log-Mek features)
shared_dim = 32 #32 shared hidden features that feed into different task heads

shared_speech_encoder = nn.Linear(in_features, shared_dim, bias=False)

#Define two separate, lightweight task heads representing different language outputs
transcription_head = nn.Linear(shared_dim, 5) #Predicts 5 phonetic text characters
translation_head = nn.Linear(shared_dim, 5)#Predicts 5 translated words

#simulate a batch of incoming audio frames
audio_input = torch.randn(4, in_features)

#The forward pass and task loss calculation
#Run audio through the shared core model
shared_features = shared_speech_encoder(audio_input)

#Pass the shared features to both the independent tasks
transcription_logits = transcription_head(shared_features)
translation_logits = translation_head(shared_features)

#Calculate the independent losses (Simulating ground-truth text matches)
loss_fn = nn.MSELoss()
loss_task1 = loss_fn(transcription_logits, torch.randn_like(transcription_logits))
loss_task2 = loss_fn(translation_logits, torch.randn_like(translation_logits))


#Calculate Isolated Task Gradients
#To perform surgery, extract the gradients of each task *individually*
#relative to the shared speech encoder weights

#Gradients for Task 1 (Transcription)
shared_speech_encoder.zero_grad()
loss_task1.backward(retain_graph=True)

#Flatten the gradiient matrix into a 1D vector for clean vector calculus operations
grad_task1 = nn.utils.parameters_to_vector([p.grad for p in shared_speech_encoder.parameters()]).clone()

#Gradients for Task 2 (Translation)
shared_speech_encoder.zero_grad()
loss_task2.backward()
grad_task2 = nn.utils.parameters_to_vector([p.grad for p in shared_speech_encoder.parameters()]).clone()

#Gradient surgery operations (PCGrad Engine)
#Check for a conflict by calculating the vector dot product : g1*g2
dot_product = torch.dot(grad_task1, grad_task2)

print("Multi-Task Speech Gradient Surgery Diagnostics")
print(f"Calculated Vector Dot Product: {dot_product.item():.4f}")

if dot_product < 0:
    print("[Conflict detected] -> Gradients are fighting. Initiating surgery")
    
    #Task 1 Surgery: Projection of grad_task1 onto the normal plane of grad_task2
    #Formula: g1_new = g1 - ((g1.g2) / ||g2||**2) * g2

    g2_norm_sq = torch.sum(grad_task2 ** 2) + 1e-12
    grad_task1_surgery = grad_task1 - (dot_product / g2_norm_sq) * grad_task2
    
    #Task 2 Surgery: Projection of grad_task2 onto the normal plane of grad_task1
    #Formula: g2_new = g2 - ((g1.g2) / ||g1||**2) * g1

    g1_norm_sq = torch.sum(grad_task1 ** 2) + 1e-12
    grad_task2_surgery = grad_task2 - (dot_product / g1_norm_sq) * grad_task1

    #Combine the post-surgery non-conflicting gradient vectors
    final_gradient_vector = grad_task1_surgery + grad_task2_surgery
    print("Success: Destructive cross-task directional vectors stripped away")
else:
    print("No conflict detected.Gradients are cooperative.")
    final_gradient_vector = grad_task1 + grad_task2

#Apply the surgically Altered Gradient to the Weights
shared_speech_encoder.zero_grad()

#Unflatten the corrected vector back into the original shape of the network parameters
#and manually inject it back into the model's grad placeholders
i = 0
for p in shared_speech_encoder.parameters():
    num_elements = p.numel()
    #extract the slice of the corrected vector belonging to this parameter layer
    surgically_corrected_slice = final_gradient_vector[i:i+num_elements]
    p.grad = surgically_corrected_slice.view(p.shape)
    i += num_elements

#create a basleine optimizer and take a step using clean, non-conflicting gradients
optimizer = optim.AdamW(shared_speech_encoder.parameters(), lr=0.01)
optimizer.step()

print(f"Final Shared Encoder Matrix Footprint: {shared_speech_encoder.weight.shape}")
print("Optimization safely completed without multi-task parameter corruption.")
