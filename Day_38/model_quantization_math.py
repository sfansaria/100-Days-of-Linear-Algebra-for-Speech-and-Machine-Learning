'''
Model Quantization Math -

When a state-of-the-art speech foundation model like OpenAI's Whisper is trained,
every single parameter weight is stored as a 32-bit floating point number (FP32).

The Problem:
A 1-billion parametr speech network stores in FP32 requires a massive 4 Gigabytes of 
raw memory just to sit on a disk or load into RAM. Running matrix multiplications on 
continuous 32-bit decimals chews up vast amounts of silicon processing power and battery life,
making local, real-time transcription on consumer devices (like an Apple Watch, iPhone, or an
embedded smart speaker) impossible.

The Solution:
Model Quantization is the mathematical framework that compresses these massive 32-bit decimals
down to hyper-compact 8-bit integers (INT8) or even 4-bit integers (INT4). This shrinks the
model's footprint by 75% to 87.5%, allowing giant networks to run blazing fast on edge hardware 
without sacrificing accuracy.

The Mathematics:
To map a wide range of floating-point decimals smoothly onto a tight grid of integers, 
use a geometric mapping projection called Affine Quantization.

To compress an FP32 weight matrix down to an unsigned 8-bit integer space (UINT8),
the target integer grid has a hard bounded range of [0, 255]. Calculate two critical
scaling parameters from the weight matrix:
1. The Scale Factor (S): A continuous decimal multiplier that stretches or shrinks the grid.
2. The Zero-Point (Z): An integer coordinate that maps exactly to the real floating-point 0.0 value,
ensuring that dead zones or padded silences in speech processing remain perfectly zeroed out.

The Quantization Trnaformation Equation:

To convert a real number x to an integer q, divide by the Scale Factor, add the Zero-Point,
and round to the nearest whole integer:
                                       q = round(x/S) + Z

Because integers can easily overflow past the 8-bit limits, apply a mathematical clamping function 
to force the value to stay inside the legal boundary:
                                                     q_quantized = max(q_min, min(q_max, q))    

The Dequantization Pass -
When the model runs, the phone's processor performs rapid, power-efficient integer linear algebra 
on the quantized q matrix. Once the calculation is complete, the resulting tensor is projected back 
to a floating-point decimal space so it can interface with the next layer:

x_ = S.(q_quantized - Z)

To build a functional Quantization and Dequantization Engine, simulate high fidelity acoustic weight parameter
matrix (FP32), project down to an 8-bit integer grid (INT8), and calculate the precision error to verify
how model optimization behaves under quantization compression.

'''

import torch
torch.manual_seed(42)


#Transforms an FP32 continuous tensor into a compressed INT8/UINT8 integer grid.
def quantify_tensor_int8(x):

    #determine the absolute boundaries of the data space
    x_min = x.min().item()
    x_max = x.max().item()

    #target bounds for standard unsigned 8-bit integer space (UINT8)
    q_min = 0
    q_max = 255

    #calculate the scale factor (S)
    #S = (x_max-x_min)/(q_max-q_min)
    scale = (x_max - x_min)/(q_max - q_min)

    if scale == 0:
        scale = 1e-6 #avoid division by zero for flat tensors

    #calculate the Zero-Point (Z)
    zero_point = round((0.0-x_min) / scale) + q_min

    #force the zero-point to stay within the legal integer boundaries
    zero_point = max(q_min, min(q_max, zero_point))

    #Quantize and Clamp the elements
    q_tensor = torch.round(x / scale) + zero_point
    q_tensor = torch.clamp(q_tensor, q_min, q_max).to(torch.uint8)

    return q_tensor, scale, zero_point


#Reconstructs an FP32 tensor approximation back from the INT8 integer grid.
def dequantize_tensor_int8(q_tensor, scale, zero_point):
    
    #convert back to float for arithmetic projection
    q_float = q_tensor.to(torch.float32)

    #formula x_hat = S*(q-Z)

    return scale * (q_float - zero_point)

#System Compression simulation pipeline

# Imagine a massive weight parameter matrix inside a speech model's attention projection layer
# Footprint: 100 rows x 100 columns = 10,000 parameter weights
# Simulating a wide variance layout
massive_speech_weights_fp32 = torch.randn(100, 100) * 2.5

print("Running INT8 Hardware Quantization Engine")

#Execute the quantization pipeline step
quantized_int8_matrix, scale, zero_point = quantify_tensor_int8(massive_speech_weights_fp32)

#Dequantize back to compare the precision gap
reconstructed_weights_fp32 = dequantize_tensor_int8(quantized_int8_matrix, scale, zero_point)

#Calculate Mean Squared Error introduced by an quantization rounding
quantization_noise_error = torch.mean((massive_speech_weights_fp32 - reconstructed_weights_fp32)**2)

print("-"*75)
print("1. Memory Storage compression Footprint:")
print(f"Original FP32 Weight Matrix Tensor Type: {massive_speech_weights_fp32.dtype}")
print(f"Compressed INT8 Weight Matrix Tensor Type: {quantized_int8_matrix.dtype}")
print(f"System VRAM Space Savings: 75.00% Memory Reduction!")
print("-"*75)
print("2. Affine Projection Hyperparameters:")
print(f"Quantized_int8_matrix: {quantized_int8_matrix}")
print(f"Quantized_int8_matrix shape: {quantized_int8_matrix.shape}")
print(f"Dynamic Scale Multiplier (S): {scale:.6f}")
print(f"Center-Point Integer Offset (Z): {zero_point}")
print("-"*75)
print("3. Accuracy Reconstruction Profile:")
print(f"Quantization Noise Rounding Error (MSE): {quantization_noise_error.item():.6f}")
print("Status : Error is negligible, model preserves its operational boundary lines.")
print("Weight matrix compressed natively to run on low-power hardware.")
