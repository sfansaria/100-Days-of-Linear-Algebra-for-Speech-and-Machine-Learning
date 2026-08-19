'''
In speech machine learning, for training Multi-Task Speech models, Gradient Surgery technically known
as Projecting Conflicting Gradients (PCGrad), a mathematical conflict-resolution technique is used.

A modern speech foundation model (like OpenAI’s Whisper) is trained to perform multiple tasks simultaneously
using the exact same shared neural network parameters:
- ASR (Automatic Speech Recognition): Transcribing what was said.
- Translation: Translating the spoken audio into another language.
- VAD (Voice Activity Detection): Detecting where speech starts and background noise ends.

Lets consider that drama director needs the actor to execute two separate task at the same millisecond.
The actor must speak in a thick, accurate french accent and must deliver the lines with aggression, angry tone.

The director reviews the scene and gives two advice simlutaneously.
number 1- To sound angry, tighten the jaw, grit the teetch, and scream. (Gradient for Task 1)
number2 - To maintain the french accent, relax the jaw, loos the mouth and gracefully shape the vowels. (Gradient for Task 2)
These two instruction completely contradict each other. If the actor tries to perform both the tasks blindly, their brain will
freze up. If the one task is performed the other wont. If they try to perform both then they cancel each other and they perform horribly.

In machine learning this is called Gradient Interference. When a model calculates the calculus gradients for Task 1 (Transcription)
and Task 2 (Translation), the two vectors point in  opposite directions, ripping the model's parameters and delaying its learning.

In order to solve this , Gradient Surgery (acting as smart director) is applied, which inspects the vectors before they hit the 
actor's brain (the model's weights)

By calculating the dot product of the vectors, the nature of the vectors are knonw whether they are fighting. 
If the dot product is negative (g_i . g_j < 0), this mathematically signifies that tasks acting destructive to one another.

PCGrad algorithm instead of adding the vectors, it projects the gradient of Task 1 onto the normal plane (perpendicular space)
of Task 2. This mathematically strips away the specific directional components that are actively sabotaging the Task 2, leaving
only the components that are healpful or neutral. The director now tells the actor to keep the intense loudness and raw energy
 of the anger, but do not grit the teeth so that the jaw remains loos enough to hit the French vowel shapes 

Without Gradient Surgery, multi-task speech models suffer from negative transfer, where adding a translation capability actively
damages the model's baseline transcription accuracy.By running PCGrad, the optimization algorithm ensures that when the network 
updates its attention layers to better understand contextual translations, it does not accidentally overwrite or destroy the 
foundational acoustic feature boundaries (like spotting short consonants or silence gaps) that it needs for basic speech 
recognition. It allows the model to learn multiple speech behaviors smoothly in tandem.

'''