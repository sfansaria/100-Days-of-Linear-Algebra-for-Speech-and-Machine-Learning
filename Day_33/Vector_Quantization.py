'''
Continuous Audio Waveforms are chaotic and complex, making them difficult
to analyse and process as they are infinitely variable. However, Text models like 
GPT excel because they process discrete, distinct tokens (word/characters).
To make speech generation  efficient, transform the continuous audio spectrograms 
into discrete ""acoustic words" or tokens. This process is called Vector Quatization (VQ).

When listening to a voice of person speaking, instead of saving the every single microscopic
decimal point of the their Mel-spectrogram, you can save  it in a massive internal dictionary called a codebook.

-> The codebook is a matrix containing a fixed number of vectors (example: 1024 vectors), where 
each row represents a "prototype" acoustic sound shape (like a clean vowel, a sharp breatthe, or a nasal sound).

-> For every incoming audio frame, the model performs a geometric search across the dictionary.
It finds the closest matching/ closest to the current audio frame vector, finds it quickly and 
outputs that dictionary index 

Continuous Audio Feature Frame       Geometric Search              The Codebook Matrix
      [ 0.72, -1.2, 0.44... ]    ──►  Find Closest Row  ──►  Row 412: [ 0.70, -1.1, 0.40... ]
                                                                       ▼
                                                             Output Discrete Token: 412


By doing this, a continuous 1-second vocal recording is compressed into a clean list of whole numbers:
[412, 12, 994, 87]. A standard language transformer can now predict speech just like it predicts words in a sentence.


'''