'''
LoRA (Low-Rank Adaptation)
The state-of-the-art 1-billion-parameter Speech Model (like OpenAI's Whisper) understands the general human language 
flawlessly. The task is to adapt it to recognize a highly niche regional accent or medical vocabulary.

---> The old way (Full Fine-Tuning): Modify all 1 billion parameters. The optimization algorithm must calculate, track, 
and save massive memory gradients for every single matrix layer, requiring industrial supercomputing clusters.

---> The efficient way (LoRA way): Freeze the massive original model matrix (W_0) completely. It becomes read-only memory, 
requiring zero gradient calculus tracking. Place a separate, lightweight path network next to it, called   (Delta W).

Instead of building Delta W as a massive matrix of the same size, use the SVD low-rank rule. Factorize it into two tiny matrices
multiplied together: Matrix A and Matrix B 

Original Hidden Audio Features (x)
             ├──► [ MASSIVE FROZEN MATRIX W₀ ] ──────────────┐
             │         (No gradients calculated)              ▼
             │                                              [ + ] ──► Scaled Output
             └──► [ Matrix A: in x r ] ──► [ Matrix B: r x out ] ──┘
                       (Trainable low-rank matrix parameters)

If the attention layer has 4096 inputs and 4096 outputs, full training tracks 16.7 million parameters.
Initialize LoRA with an intrinsic rank r = 4
Matrix A has a shape of 4 * 4096 = 16384 parameters
Matrix B has a shape of 4096 * 4 = 16384 parameters
Total parameters to train: 16,384 + 16,384 = 32, 768 parameters total

Optimize only 0.2 % of the parameter space, saving massive amounts of GPU VRAM while achieving nearly identical model accuracy.

lets build a functional LoRA injection block, that hooks it into an acoustic model layer, freezes the core weights, and runs
an optimization update step using the simulated audio.

'''

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


#LoRA injection layer
class LoRABlock(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=8):
        super().__init__()
        self.base_layer = base_layer
        
        #freeze the main massive acoustic network layer completely
        self.base_layer.weight.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features


        #Initialize the two low-rank factorization matrices
        #Matrix A is initialized with standard Gaussian random scaling
        self.lora_A = nn.Parameter(torch.randn(rank, in_features)*0.01)

        #Matrix B is initialized to flat zeroes so that at step 0, LoRA outputs exactly 0
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.scaling = alpha / rank
    
    def forward(self, x):
        #Path 1 the massive frozen baseline intelligence output
        frozen_out = self.base_layer(x)

        #Path 2 the lighweight low-rank adaptation trajectory path
        #x shape: [Batch, In_Features] -> Matrix A -> Matrix B
        lora_features = torch.matmul(x, self.lora_A.t())
        lora_out = torch.matmul(lora_features, self.lora_B.t())
        
        #Fuse them together
        return frozen_out + (lora_out * self.scaling)



#execute the engine pipeline
## Simulating a deep speech layer processing 80 Mel acoustic input channels

base_linear_layer = nn.Linear(80, 80, bias=False)
lora_enhanced_layer = LoRABlock(base_linear_layer, rank=4)

#Confirm parameters are isolated for optimization
optimizer = optim.AdamW(lora_enhanced_layer.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


#Simulate accent-heavy acoustic frame input and targeted text expectation vectors
audio_input_frame = torch.randn(1, 80)
target_linguistic_vector = torch.randn(1, 80)

print("Initializing Low-Rank Adaptation (LoRA) optimization step")

optimizer.zero_grad()
predictions = lora_enhanced_layer(audio_input_frame)
loss = loss_fn(predictions, target_linguistic_vector)
loss.backward()
optimizer.step()

print(f"Fine-Tuning Performance Loss Evaluation: {loss.item():.4f}")
print(f"Shared Base Layer Gradient Track Status: {base_linear_layer.weight.grad}")
print(f"Trainable Matrix A calculated Grad Norm: {lora_enhanced_layer.lora_A.grad.norm().item():.4f}")
print(f"Trainable Matrix B calculated Grad Norm: {lora_enhanced_layer.lora_B.grad.norm().item():.4f}")
print("Core speech memory remained frozen whil LoRA matrices absorbed training gradients. ")