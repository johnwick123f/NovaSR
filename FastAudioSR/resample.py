import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# --- MATH UTILS (Updated for 16-tap) ---
def get_kaiser_filter1d(kernel_size=16, cutoff=0.5):
    # Standard sinc filter with Kaiser window for anti-aliasing
    half_size = kernel_size // 2
    # Beta 4.558 is a standard choice for audio resampling
    window = torch.kaiser_window(kernel_size, beta=4.558, periodic=False)
    # 16-tap is even, so we use the 0.5 offset for center alignment
    time = torch.arange(-half_size, half_size) + 0.5
    
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)

# --- MODULES ---

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, channels=512):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.kernel_size = 16 
        
        # This matches your checkpoint key, but we will overwrite it with 16-tap math
        self.register_buffer("filter", torch.zeros(1, 1, 16))
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Generate fresh 16-tap weights for quality
            f = get_kaiser_filter1d(kernel_size=16, cutoff=0.5 / self.ratio)
            f = f * float(self.ratio)
            # Expand for grouped convolution: [channels, 1, 16]
            self.filter.copy_(f) 
            self._prepared_weight = self.filter.expand(self.channels, 1, 16).contiguous()
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared: self.prepare()
        
        # Transposed convolution is the fastest hardware-accelerated upsampler
        return F.conv_transpose1d(
            x, self._prepared_weight[:x.shape[1]], 
            stride=self.ratio, 
            padding=7,      # Correct padding for 16-tap center alignment
            output_padding=1, 
            groups=x.shape[1]
        )

class LowPassFilter1d(nn.Module):
    def __init__(self, stride=1, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.kernel_size = 16
        
        self.register_buffer("filter", torch.zeros(1, 1, 16))
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            f = get_kaiser_filter1d(kernel_size=16, cutoff=0.5 / self.stride)
            self.filter.copy_(f)
            self._prepared_weight = self.filter.expand(self.channels, 1, 16).contiguous()
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared: self.prepare()
        
        return F.conv1d(
            x, self._prepared_weight[:x.shape[1]], 
            stride=self.stride, 
            padding=8, 
            groups=x.shape[1]
        )

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, channels)
        
    def forward(self, x):
        return self.lowpass(x)
