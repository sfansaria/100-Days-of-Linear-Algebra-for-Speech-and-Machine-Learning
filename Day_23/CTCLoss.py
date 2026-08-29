'''
CTC Loss (Conectionist Temporal Classification)

When a human speaks, they do not have a constant, uniform speed. One person might say the word "CAT"
very quickly in 300 milliseconds. Another person might draw it out, saying "CAAAAT" over 1.5 seconds.

In the previous code, it was assumed an expert human labeler mapped every single millisecond frame 
to a specific phoneme target. But in real world, this is never happens. 
A dataset gives two raw elements:
1. An audio file lasting 2 seconds (200 spectral time frames)
2. A text trranscript label: "cat"

- Real Input Data Grid                  --->              Real Output Target Label
- 200 contiuous frames of sound waves   --->              "cat" (3 characters long)


The problem is how to calculate a Maximum Likelihood gradient when there is 200 time steps but only 3 text characters, 
and do not have any idea where the letters begin or end?

The solution is the mathematical magic of CTC:
There are two core concepts: 
1. Concepts A: The Blank Token (-) : CTC introduces a special null token, represented as - (blank).
This token allows the neural network to predict "nothing" or "silence" during ambinet pauses or between repeated characters.

2. Concept B: Collapsing Paths via Rules: If the model outputs a prediction at all 200 time frames, 
it might guess a raw sequence like: c - c - a a a - t t -
CTC enforces a specific collapsing function (B) that compresses this raw sequence down using two simple rules:
1. Merge sequential duplicate characters (for example a a a becomes a).
2. Remove all blank tokens (-).
when the rules are applied to c - c - a a a - t t -, it collapses perfectly down to "cat".

3. Concept C: Summing all the possible truths:
As the single correct alignment is not know, CTC takes every possible path that could ever collapse into "cat 
and sums their probabilities together.

If there are hundreds of differenr frame variations that merge down to "cat", 
the overall likelihood of the word is the sum of the probabilities of all those paths combined:
P("cat" | Audio) = (pie  belongs to inverse B ("cat"))summation P(pie|Audio)

The loss function is the Negative Log-Likelihood of that combined sum.
By running this through the optimizer, the network automatically learns to push up the probabilities of 
all paths that yield the right word, adjusting itself to different human speech speeds completely on its own.

To calculate the massive sum efficiently without checking billlions of paths one by one, CTC uses Dynamic Programming 
(specifically, a Forward-Backward algorithm similar to Hidden Markov Models).

'''

import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#Define the sequence dimensions
num_time_frames = 40 #the model predicts the character 40 times over the audio clips
num_mel_features = 80 #80 input acoustic features

#target alphabet: [0: /blank/, 1: 'c', 2: 'a', 3: 't']
num_classes = 4

#simulate raw audio feature matrix: Tensor shape [Time, Batch, Features]
#PyTorch CTCLoss natively prefers Time as the first dimension
simulated_audio = torch.randn(num_time_frames, 1, num_mel_features)

#Sequential Acoustic Model
class CTC_AcousticModel(nn.Module):
    def __init__(self, in_features, classes):
        super().__init__()
        self.network = nn.Linear(in_features, classes)
        #Log-Softmax is mathematically required because CTCLoss expects log probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        #Input shape: [Time, Batch, Features]
        logits = self.network(x)
        return self.log_softmax(logits)
    

model = CTC_AcousticModel(num_mel_features, num_classes)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

#setup the alignment free Target label
#pass a clean array containing just the characters for "cat" (Indices: 1, 2, 3)
#which frame matches which letter is not specified

#This represents ['c', 'a', 't']
true_text_target = torch.tensor([1, 2, 3], dtype=torch.long)

#Meta-tensors indicates the sequence lengths 
#Length of audio (40)
input_lengths = torch.tensor([num_time_frames], dtype=torch.long)
              
#Length of text target (3)

target_lengths = torch.tensor([3], dtype=torch.long)

#Instantiate the CTC Loss Function 
#zero_infinity=True,  safely zeroes out any illegal path probabilities that hit infinity

ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
print("Running the Alignment-free CTC optimization Loop")

for epoch in range(5):
    optimizer.zero_grad()

    #Forward Pass: Generate the Log-Probability sequence matrix
    #output shape: [Time steps (40), Batch (1), Classes (4)]
    logs_probs_matrix = model(simulated_audio)

    #Compute the CTC Dynamic Programming Loss which automatically find 
    # and sum all 40-frame paths that collapse into 3-token target
    loss = ctc_loss_fn(logs_probs_matrix, true_text_target, input_lengths, target_lengths)

     #run calculus and optimize the weights
    loss.backward()
    optimizer.step()

    print(f"Iteration {epoch+1}:")
    print(f"Calculated CTC Loss: {loss.item():.4f}")

    # Let's peek at what the model is predicting at the very first frame (t=0)
    # Transforming log-probs back to standard probabilities via exponentiation (exp)
    frame_0_probs = torch.exp(logs_probs_matrix[0, 0]).tolist()
    print(f"->  Frame 0 probs -> [-]: {frame_0_probs[0]:.3f}, [c]: {frame_0_probs[1]:.3f}, [a]: {frame_0_probs[2]:.3f}")

print("The model is learning text representations directly from unaligned audio")


'''
The training script executes perfectly without any hardcoded frame-by-frame text indices.
By passing the raw sequence through nn.CTCLoss, the algorithm sets up a trellis grid behind the scenes. 
It automatically calculates the partial derivatives across every single valid alignment variation simultaneously. 
This math is exactly what allowed modern voice systems to scale up to millions of hours of unaligned conversational recordings.

'''
