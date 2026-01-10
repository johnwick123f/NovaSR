import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def get_kaiser_filter1d(kernel_size, cutoff):
    half_size = kernel_size // 2
    window = torch.kaiser_window(kernel_size, beta=4.558, periodic=False)
    time = torch.arange(-half_size, half_size) + 0.5
    f = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (f / f.sum()).view(1, 1, kernel_size)

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, channels=512):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        
        # Keep at 12 to satisfy your pretrained state_dict loader
        self.register_buffer("filter", torch.zeros(1, 1, 12))
        
        # Execution weight: [In_channels, Out_per_group, K]
        # For ConvTranspose1d depthwise, Out_per_group must be 1.
        self.register_buffer("f_fast", torch.zeros(channels, 1, 16), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            # Force 16-tap math
            f = get_kaiser_filter1d(16, 0.5 / self.ratio) * self.ratio
            # Shape for Transposed Conv (depthwise): [C, 1, 16]
            self.f_fast.copy_(f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared: self.prepare()
        C = x.shape[1]
        # Groups must equal C for depthwise. Weight is [C, 1, 16].
        return F.conv_transpose1d(
            x, self.f_fast[:C], 
            stride=self.ratio, padding=7, output_padding=1, groups=C
        )

class LowPassFilter1d(nn.Module):
    def __init__(self, stride=1, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        
        # Match checkpoint size 12
        self.register_buffer("filter", torch.zeros(1, 1, 12))
        
        # Execution weight: [Out_channels, In_per_group, K]
        # For Conv1d depthwise, In_per_group must be 1.
        self.register_buffer("f_opt", torch.zeros(channels, 1, 16), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            f = get_kaiser_filter1d(16, 0.5 / self.stride)
            self.f_opt.copy_(f.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared: self.prepare()
        C = x.shape[1]
        # Weight shape [C, 1, 16] works for groups=C
        return F.conv1d(
            x, self.f_opt[:C], 
            stride=self.stride, padding=8, groups=C
        )

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, channels)
    def forward(self, x):
        return self.lowpass(x)
