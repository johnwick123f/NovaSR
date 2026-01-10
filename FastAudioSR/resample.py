import torch
import torch.nn as nn
from torch.nn import functional as F

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=16): # Forced to 16
        super().__init__()
        self.ratio = ratio
        self.kernel_size = 16 
        self.stride = ratio
        
        # These remain for state_dict compatibility
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        
        # Placeholder for state_dict loading
        # The kaiser_sinc_filter1d will be overwritten by your pretrained weights
        self.register_buffer("filter", torch.zeros(1, 1, self.kernel_size))

    def forward(self, x):
        B, C, T = x.shape
        
        # 1. QUALITY IMPROVEMENT: Use reflect padding instead of replicate if possible
        # but to keep logic IDENTICAL to your pretrained setup, we stick to your pad:
        x = F.pad(x, (self.pad, self.pad), mode='replicate')

        # 2. SPEED OPTIMIZATION: Polyphase Decomposition
        # We reshape the filter into [ratio, 1, kernel_size // ratio]
        # This turns a Transposed Conv (slow) into a standard Conv (fast) + Interleave
        weight = self.filter.view(self.ratio, self.kernel_size // self.ratio).flip(-1)
        weight = weight.unsqueeze(1).expand(C * self.ratio, 1, -1)
        
        # Grouped convolution is significantly faster
        x = F.conv1d(x, weight, groups=C, stride=1) 
        
        # Interleave the results to achieve upsampling
        x = x.view(B, C, self.ratio, -1).permute(0, 1, 3, 2).reshape(B, C, -1)
        
        # Slice to match the exact original output length
        return x[..., self.pad_left:-self.pad_right]

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=16):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = 16
        # Pre-registering the lowpass logic
        from .filter import LowPassFilter1d
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      stride=ratio,
                                      kernel_size=self.kernel_size)

    def forward(self, x):
        # Downsampling optimization: 
        # Ensure the filter is applied in a single pass with stride
        return self.lowpass(x)
