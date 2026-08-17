'''
Let us imagine as a sound engineer training an Automatic Speech Recognition (ASR) system. 
Test the model in a perfectly silent, isolated audio booth using high-end studio microphones.
Tweak the weights until the model hits a 0% error rate.

If the model is trained into a Sharp Minimum, the mathematical settings are hyper-customised 
only for the unique echo, acoustic dryness, and dimensions of that exact room. 
The second a user tries the model in a noisy car or a bustling cafe, the acoustic landscape shifts slightly, 
and model's accuracy completely drops.

The standard optimizer like AdamW only sees where is the lowest error right now, this creates a fragile narrow pit.
But Sharp Aware Minimization (SAM) is like a strict, adversarial audio stress-tester, Before updating the weight,
it performs a two-step lookahead.
1. SAM introduces distortion in current settings by injecting an adversarial burst of background noise and vocal jitter.
2. If this noise creates a catastrophic failure, SAM rejects this spot, even if the current error is zero. 
It intentionally searches for a flat minimum (a wide , robust valley) where even if the background hums, room reflections, or 
unexpected accents shift the landscape, the model's accuracy remains stable.

lets build a SAM Optimizer Loop wrapper. Create a simulation of an acoustic layer and show how SAM forces the model 
to look ahead at the "worst-case neighbourhood" before updating the weights.

'''

import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#simulate the speech setup with 13 MFCC features and 1 voice activity output (0=Silence, 1=Speech)
model = nn.Linear(13, 1)
loss_fn = nn.BCEWithLogitsLoss()

#simulate a batch of audio features and target labels
x_audio = torch.randn(8, 13)
y_target = torch.randint(0, 2, (8, 1)).float()

#SAM optimization
class SAMwrapper:
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        #rho - the radius of the 'neighborhood' to check the sharpness.
        self.params = list(params)
        self.base_optimizer = base_optimizer_cls(self.params, **kwargs)
        self.rho = rho
    
    @torch.no_grad()

    def first_step(self):
        #Adversarial perturbation 'e_w' to update the weights according to the worst case local spot.
        
        #collective norm of all the gradients across layers
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in self.params if p.grad is not None]))
        
        #scaling the factor based on the neighborhood size (rho)
        scale = self.rho / (grad_norm + 1e-12)

        for p in self.params:
            if p.grad is None: 
                continue
            #e_w is the worst case noise direction for this parameter
            e_w = p.grad * scale
            p.add_(e_w) #update the actual weight to the distorted spot
            p.adv_noise = e_w #cache it for later
    
    @torch.no_grad()
    def second_step(self):
        #Undo the adversarial nudge and update weights using the gradient calculated from the rugged spot
        for p in self.params:
            if p.grad is None:
                continue

            p.sub_(p.adv_noise) #Restore the weights to their original spot
        
        self.base_optimizer.step()
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()



sam_optimizer = SAMwrapper(model.parameters(), optim.AdamW, lr=0.01, rho=0.05)

predictions = model(x_audio)
loss_original = loss_fn(predictions, y_target)

loss_original.backward() #baseline gradient direction

sam_optimizer.first_step() #applying the adversarial neighborhood 

#recalculate loss at the distorted spot
sam_optimizer.zero_grad() 
loss_adversarial = loss_fn(model(x_audio), y_target)
loss_adversarial.backward()#gradient at the worst case spot

#stepp backward out of the noise and execute the smart update
sam_optimizer.second_step()
sam_optimizer.zero_grad()
print(f"baseline clean loss: {loss_original.item():.4f}")
print(f"Local Adversarial Loss: {loss_adversarial.item():.4f}")

