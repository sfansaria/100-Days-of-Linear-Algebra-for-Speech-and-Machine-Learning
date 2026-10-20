'''
Voice Token-Language Model operates as an Autoregressive Decoder.
It inputs of text characters (the script) and learns to predict a 
multi-layered matrix grid of audio codes over time.

[Text Script] ➔ Text Embedding ───┐
                                   ▼
[Past Audio Codes] ➔ RVQ Embedding ➔ [ Autoregressive Transformer Engine ] ➔ [ Predict Next Audio Frame Grid ]

The Residual Vector Quantization (RVQ) preserves fine vocal nuances, 
each time frame step does not contain a single token.
It contains a vertical layer of codes (Codebooks 1, 2, and 3).

To predict these code layers autoregressively without blowing up sequence length parameters,
researchers use Flattened Delays / Delay patterns.
The Math : Instead of predciting all the 3 layers at the exact same millisecond,
the inputs are shifted by a 1 step timeline delay.
The Result: The model can use standard, fast causal attention loops to calculate
exactly how a newly generated breath sound depends on both the text vowels and the 
vocal pitch token generated a fraction of a millisecond prior.

Build a functional Autoregressive Voice Token Transformer Decoder Layer.
Pass a simulated text script alongside a historical grid of multilayered
RVQ tokens, run a causal fusion pass, and predict the next multi-head
audio code output array.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class VoiceTokenLanguageModel(nn.Module):
    def __init__(self, vocab_size=30, codebook_size=512, num_stages=3, d_model=192):
        super().__init__()
        self.num_stages = num_stages
        self.d_model = d_model

        #Text script embedding layer
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        #RVQ Audio Codebook Embeddings (1 independent vector matrix per stage)
        self.audio_embeddings = nn.ModuleList([nn.Embedding(codebook_size, d_model) for _ in range(num_stages)])

        #core causal Transformer Decoder Layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True)

        #Hierarchial Output heads mapping back to discrete token choices
        self.output_heads = nn.ModuleList([nn.Linear(d_model, codebook_size) for _ in range(num_stages)])

    def forward(self, text_tokens, past_audio_code_grid):
        #text_tokens shape: [Batch(1), Text_Len(10)]
        #past_audio_code_grid shape: [Batch(1), Num_Stages(3), Time_Steps(16)]
        b, num_stages, time_steps = past_audio_code_grid.shape

        #step 1: Project text script into continuous language spaces
        text_feats = self.text_embedding(text_tokens) #[b, 10, d_model]

        #step 2: Project and fuse the multi-layer RVQ codebooks
        #accumulate the vector spaces and across the 3 hierarchial stages element-wise
        audio_feats = torch.zeros(b, time_steps, self.d_model, device=past_audio_code_grid.device)
        for stage_idx in range(self.num_stages):
            stage_codes = past_audio_code_grid[:, stage_idx, :] #[b, 16]
            audio_feats += self.audio_embeddings[stage_idx](stage_codes)
        
        #step 3: Generate a Causal attention Matrix Mssk
        # Forces the model to only look backward, ensuring true generation capability
        mask = nn.Transformer.generate_square_subsequent_mask(time_steps, device = past_audio_code_grid.device)


        #step 4: execute the cross-attention fusion pass
        #Target = Audio features (with causal mask)
        #Memory = Text condition script features
        fused_hidden_states = self.decoder_layer(tgt=audio_feats, memory=text_feats, tgt_mask=mask)


        #step 5 Predict the next token  probabilities across ALL codebook stages in parallel

        stage_logits = []

        for stage_idx in range(self.num_stages):
            logits = self.output_heads[stage_idx](fused_hidden_states)
            stage_logits.append(F.log_softmax(logits, dim=-1))
        
        # Stack into final structured grid: [Num_Stages(3), Batch(1), Time(16), Codebook_Choices(512)]
        return torch.stack(stage_logits, dim=0)
    

#System Inference simulation

#simulate a 10-word input text script
simulated_script = torch.tensor([[1, 5, 12, 4, 8, 22, 19, 3, 15, 2]], dtype=torch.long)

#simulate 16 steps of historical audio data across  3 RVQ codebook channels
simulated_past_audio = torch.randint(0, 512, (1, 3, 16))

voice_lm = VoiceTokenLanguageModel()
output_probability_lattice = voice_lm(simulated_script, simulated_past_audio)

print("="*75)
print("AUTOREGRESSIVE AUDIO LANGUAGE MODEL")
print("="*75)
print(f"Input Text Script Token Sequence:      {list(simulated_script)} (10 tokens)")
print(f"Input Text Script Token Sequence shape :      {list(simulated_script.shape)} (10 tokens)")
print(f"Input Hierarchical Past Audio Grid:    {list(simulated_past_audio)}")
print(f"Input Hierarchical Past Audio Grid shape:    {list(simulated_past_audio.shape)}")
print("Dimensions: [Batch, 3 RVQ Layers, 16 Time Steps]")
print("-"*75)
print(f"OUTPUT PROBABILITY LATTICE FOOTPRINT: {list(output_probability_lattice)}")
print(f"OUTPUT PROBABILITY LATTICE FOOTPRINT: {list(output_probability_lattice.shape)}")
print(" Layout: [3 Stages, Batch=1, 16 Time Steps, 512 Dictionary Choices]")
print(" Causal masks successfully forced temporal synthesis.")
print("="*75)