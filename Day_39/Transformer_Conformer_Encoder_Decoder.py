'''
The Global Structural Architecture: 
Whisper is a multi-task Sequence-to-Sequence (Seq2Seq) Transformer.
It maps a variable-length continuous acoustic matrix directly to a variable-length text token matrix.

The model is divided into two primary engines:

[Raw Audio] ➔ 2D Conv Subsampling ➔ Positional Encoding ➔ Stacked Self-Attention Blocks ➔ [Encoder Hidden States]
                                                                                                  │
                                                                                                  ▼
[Prompt Tokens] ➔ Text Embedding ➔ Masked Self-Attention ➔ Cross-Attention (Q from Text, K,V from Audio) ➔ [Text Output]


The Encoder Engine - 
The input is an 80-channel Log-Mel Spectrogram Matrix.
1. Time Compression: It passes through two 2D Convolution layers with a stride of 2.
This compresses the timeline by exactly 4x, protecting downstream layers from quadratic complexity growth.
2. Time Ordering: Sinusoidal Positional Encodings are added element-wise to stamp each downsampled audio
slot with a unique timestamp vector.
3. Contextual Extraction: The matrix passes through a stack of standard Transformer Encoder blocks.
These use Multi-Head Self-Attention to evaluate audio context across the entire timeline simultaneously, 
outputting highly informative Encoder Hidden States.

The Decoder Engine:
The decoder is auto-regressive, it predicts the text transcription token-by-token.
It handles two distinct attention tasks:
1. Masked Self-Attention: Evaluates the history of text tokens that the model has already typed out,
preventing it from looking ahead into future words.
2. Multi-Head Cross-Attention (The Core Fusion): This is where audio meets language.
The decoder maps its linguitic Queries (Q) to search through the acoustic Keys (K) and Values (V)
provided directly by the Encoder Hidden States. 
It translates vocal pitches and vowel weights into pure textual probabilities.

Simulation of the Whisper End-to-End Encoder-Decoder pipeline uisng tensor.
Track the matrix dimension shifts explicitly as data moves audio waves to text tokens.

'''

import torch 
import torch.nn as nn
import math

torch.manual_seed(42)

#The Whisper encoder engine
class WhisperEncoder(nn.Module):
    def __init__(self, in_features=80, encoder_dim=256):
        super().__init__()
        
        #4x Downsampling  Audio Front-End
        self.conv_subsample = nn.Sequential(
                                           nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
                                           nn.ReLU(),
                                           nn.Conv2d(64, encoder_dim, kernel_size=3, stride=2, padding=1),
                                           nn.ReLU()   
                                          )
        
        #Linear Projection to align dimensions after flattening frequency bins
        #20 remains from 80 after 2 strides
        self.proj = nn.Linear(encoder_dim*20, encoder_dim)

        #Positional Encoding Buffer
        self.register_buffer('pe', torch.zeros(1, 1000, encoder_dim))
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoder_dim, 2).float() * (-math.log(10000.0) / encoder_dim))
        self.pe[:, :, 0::2] = torch.sin(position*div_term)
        self.pe[:, :, 1::2] = torch.cos(position*div_term)


        #Self-Attention Layer Block
        self.attn = nn.MultiheadAttention(embed_dim=encoder_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(encoder_dim)
    

    def forward(self, x):
        #x shape : [Batch, Time_Frames(100), Mel-features(80)]
        b, t, f = x.shape
        x_4d = x.unsqueeze(1) #Add channel axis for Conv2D

        out = self.conv_subsample(x_4d) #[b, 256, t/4, f/4]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, out.shape[2], -1) #Flatten frequency
        out = self.proj(out) #Project to [b, t/4, 256]

        #Add timestamps and run Self-Attention context extractor
        out = out + self.pe[:, :out.shape[1], :]
        attn_out, _ =  self.attn(out, out, out)
        return self.norm(attn_out)


#Whisper Decoder Cross-Attention Engine
class WhisperDecoder(nn.Module):
    def __init__(self, vocab_size=4, decoder_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim)

        #Text Self-Attention and Audio-Text Cross-Attention Layers
        self.self_attn = nn.MultiheadAttention(embed_dim=decoder_dim, num_heads=4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=decoder_dim, num_heads=4, batch_first=True)

        self.output_head = nn.Linear(decoder_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, text_tokens, encoder_hidden_states):
        #text_tokens shape: [Batch, Text_Len(5)]
        #encoder_hidden_states shape: [Batch, Compressed_Audio_Time(25), Decoder_Dim(256)]
        #Embed token indices into continuous language spaces
        tgt = self.token_embedding(text_tokens)
        
        #Pass A: Masked Self-Attention over text text history
        #Using a causal mask grid would happen here in full networks
        text_context, _ = self.self_attn(tgt, tgt, tgt)

        #Pass B: Cross-Attention Core Fusion
        #Queries (Q) originate from text context
        #Keys (K) and Values (V) arise from audio encoder states
        fused_context, _ = self.cross_attn(query=text_context, key=encoder_hidden_states, value=encoder_hidden_states)

        #Predicts character vocabulary logits
        logits = self.output_head(fused_context)
        return self.log_softmax(logits)
    


#End to End System Simulation 

# Simulate 1 second of dense audio features (100 frames × 80 Mel bins)
simulated_audio = torch.randn(1, 100, 80)

# Simulate 5 historical prompt text tokens [e.g., <|startoftranscript|> "
simulated_text = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.long)

whisper_encoder = WhisperEncoder()
whisper_decoder = WhisperDecoder()

audio_hidden_features = whisper_encoder(simulated_audio)
text_output_probabilities = whisper_decoder(simulated_text, audio_hidden_features)

print("="*75)
print(" SYSTEM ARCHITECTURE BLUEPRINT (OpenAI Whisper)")
print("="*75)
print(f"Input Acoustic Wave Matrix Shape:    {list(simulated_audio.shape)} (Time x Features)")
print(f"simulated_audio: {list(simulated_audio)}")
print(f"ENCODER HIDDEN STATES MATRIX:  {list(audio_hidden_features)}")
print(f"ENCODER HIDDEN STATES MATRIX SHAPE:  {list(audio_hidden_features.shape)}")
print("   -> Timeline verified: Compressed 4x by convolutions (100 -> 25 steps).")
print("-"*75)
print(f" Input Text Decoder Prompt Tokens:    {list(simulated_text)}")
print(f" Input Text Decoder Prompt Tokens:    {list(simulated_text.shape)} (5 input words)")
print(f" FINAL TEXT VOCABULARY OUTPUT GRID:   {list(text_output_probabilities)}")
print(f" FINAL TEXT VOCABULARY OUTPUT GRID:   {list(text_output_probabilities.shape)}")
print("   -> Interpretation: Fused 3D lattice mapping 5 target token probabilities.")
print("===========================================================================")
