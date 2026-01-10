# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# --- MATH UTILS ---
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.: beta = 0.1102 * (A - 8.7)
    elif A >= 21.: beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else: beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size) + 0.5) if even else (torch.arange(kernel_size) - half_size)
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)

# --- FUSED KERNEL (Restored to 12-tap Alignment) ---
@torch.jit.script
def _polyphase_upsample_fused(x: Tensor, weight: Tensor, ratio: int):
    # Original padding for 12-tap center alignment
    x = F.pad(x, (2, 3))
    out = F.conv1d(x, weight, groups=x.shape[1], stride=1)
    
    B, C_out, L = out.shape
    C = x.shape[1]
    out = out.view(B, C, ratio, L).transpose(2, 3).reshape(B, C, -1)
    # Original slice for 12-tap alignment
    return out[..., 2:-2]

# --- MODULES ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --- OPTIMIZED MODULES ---

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, channels=512):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.kernel_size = 16  # Forced to 16
        
        # State-dict compatibility: Keep the original buffer names
        self.register_buffer("filter", torch.zeros(1, 1, 16))
        
        # We replace the manual fused kernel with a native Transposed Conv
        # This is the "fast" path. We use groups=channels for depthwise-style speed.
        self.up_conv = nn.ConvTranspose1d(
            channels, channels, 
            kernel_size=16, 
            stride=ratio, 
            padding=7,     # Adjusted for center alignment with 16-tap
            output_padding=1, 
            groups=channels, 
            bias=False
        )
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Align the pretrained 'filter' weights to the Transposed Conv format
            # Weights in ConvTranspose1d are [In, Out/Groups, K]
            w = self.filter.expand(self.channels, 1, 16)
            self.up_conv.weight.copy_(w)
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared and not self.training:
            self.prepare()
        
        # Native Transposed Conv is much faster than manual polyphase + transpose/reshape
        return self.up_conv(x)

class LowPassFilter1d(nn.Module):
    def __init__(self, stride=1, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.kernel_size = 16 # Forced to 16
        
        # State-dict compatibility
        self.register_buffer("filter", torch.zeros(1, 1, 16))
        
        # Optimized path using a standard grouped convolution
        self.conv = nn.Conv1d(
            channels, channels, 
            kernel_size=16, 
            stride=stride, 
            padding=8, # Padding for 16-tap symmetry
            groups=channels, 
            bias=False
        )
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            w = self.filter.expand(self.channels, 1, 16)
            self.conv.weight.copy_(w)
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared and not self.training:
            self.prepare()
        # Direct convolution call is faster than functional F.conv1d with slicing
        return self.conv(x)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, channels)

    def forward(self, x):
        return self.lowpass(x)
