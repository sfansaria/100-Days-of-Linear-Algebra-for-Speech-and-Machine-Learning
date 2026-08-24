'''
The transition from static statistics to sequential deep learning, while moving from basic math to Automatic Speech Recognition(ASR), 
the inputs and outputs change from single scalar values to high-dimensional arrays (vectors) that vary over time.
In real life, a single slice of time contains 25 milliseconds which contains much more information than a just a pitch.
It contains features like volums, texture, vocal resonances, and tone. To capture this, the frame is converted into a vector of features
traditionally containing 80 Mel-Filterbanks coefficients.
In basic way, the model outputted a single normal curve trying to predict that one pitch number.
In real ASR system, the model need to categorize what sound is being spoken out of an entire alphabet or phonetic dictionary.
If the model is designed to recognize english, it needs to evaluate all the possible phonemes (the individual units of sound
that make up words like /ah/, /t/, /s/, /sh/)
If the dictionary has 40 distinct phonemes, the output of the neural network layer for that single frame is a vector of 40
numbers (called logits). Pass this vector through a softmax function, transforming it into a true probability distribution
vector:
     output_vetor = [P(/ah/), P(/t/), P(/s/), ...., P(/sh/)]

Each slot in the vector holds a value between 0.0 and 1.0, and the entire vector sums up to exactly 1.0

Sequential Matrix Chain:
Human speech is continuous. A speaker does not just say one frame, they speak a timeline of frames.
If the user says a word "CAT" and the recording lasts for 1 second, the feature extractor chops that audio into 
a sequence of 100 individual time steps.
The input becomes a sequence matrix of shape (100 frames, 80 audio features)
The neural network processes each frame sequentially, 
using matrix calculus to output a matching Probability Sequence Matrix of  shape (100 frames, 40 phonemes).

Imagine a human expert looks at the 1-sec recording of the "CAT" and tells which phone belongs to which frame.
frame 1 to 30 = /k/
frame 31 to 70 = /ae/
frame 71 to 100 = /t/

For applying the Maximum Likelikhood Estimation, the optimizer looks at the model's output matrix and
extracts the specific probability assigned to the correct sound at every single frame.

at frame 1 , the model gave the probability to /k/, P(/k/) = 0.40
at frame 31, the model gave the probability to /ae/, P(/ae/) = 0.15
at frame 71, the model gave the probability to /t/, P(/t/) = 0.60

The total sequence loss is the sum of the negative logs of those exact correct targets across the entire time chain.

Sequence Loss = - t=1to100 summation(log P_t(Correct Phoneme))

Otimization Algorithm like AdamW or SAM calculates the partial derivatives of this sequential sum, working backward
through time to warp the model weight's. This forces the correct phoneme probabilities to shoot up toward 1.0 and
suprresse the incorrect guesses down to 0.0.

The research problem: This sequential expansion works beautifully if and only if the exact time of the alignment is known,
precisely which millisecond transitions from /k/ to /ae/.
The real-world collected data does not conatins any of this. A researcher only gets an audio file and a text label like 
"cat". But dont know where the letters lines up. CTC is introduced to solve this problem, which uses advanced probability 
to calculate this MLE Sequence without knowing the frame alignments. 

'''
