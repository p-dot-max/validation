
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import torch.nn.functional as F

'''
To Do

Implement Efficent Zpool
    |-> Average Pooling 
    |-> Max Pooling

Use SiLU Activation Function from nn module 

Implement channel Attention & Spatial Attention 


Use featurs of ECA Attention mechanism
Incorporate Channel, Spatial and ECA Attention

Incorp with RESCBAM bottleneck

'''

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        print(f"Concatenating along dimension {self.d}")
        for i, tensor in enumerate(x):
            print(f"Tensor {i} size: {tensor.size()}")
        return torch.cat(x, self.d)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Zpool(nn.Module):
    def __init__(self, kernel_size=3, stride=2):
        super(Zpool, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding=kernel_size // 2)  # Ensure alignment
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding=kernel_size // 2)
        self.concat = Concat(dimension=1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        # print(f"AvgPool output shape: {avg_out.shape}")
        # print(f"MaxPool output shape: {max_out.shape}")
        return self.concat([avg_out, max_out])


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction=16):
        super(ChannelAttention, self).__init__()
        self.z_pool = Zpool()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels*2, channels // reduction, 1, bias=False)  # Reduce dimensionality
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)  # Back to original channels
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_pooled = self.z_pool(x)  # Output shape: [batch, channels * 2, h // 2, w // 2]
        attention = self.fc1(z_pooled)
        attention = self.act(attention)
        attention = self.fc2(attention)

        # Resize attention to match x's spatial dimensions
        attention = F.interpolate(attention, size=x.shape[2:], mode="bilinear", align_corners=False)

        return x * attention.sigmoid()


class SpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 7]):
        super(SpatialAttention, self).__init__()
        self.attention_blocks = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Average pooling across spatial dimensions
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # Max pooling across spatial dimensions
        combined = torch.cat([avg_pool, max_pool], dim=1)  # Concatenate along channel dimension

        attention_maps = [self.act(block(combined)) for block in self.attention_blocks]
        attention = sum(attention_maps)  # Sum attention maps from multiple kernels
        
        return x * attention.sigmoid()  # Apply attention


# class ECA_layer(nn.Module):
#     def __init__(self, channel, k_size=3):
#         super(ECA_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(channel, 1, kernel_size=k_size, padding=(k_size-1) // 2, bias=False)
#         self.bn = nn.BatchNorm1d(1)
#         self.silu = nn.SiLU()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))
#         y = self.bn(y)  # Apply batch normalization
#         y = self.silu(y).transpose(-1, -2).unsqueeze(-1)
#         return x * y.expand_as(x)


class ECA_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm1d(1)
        self.silu = nn.SiLU()

    def forward(self, x):
        y = self.avg_pool(x)  # Shape: [batch_size, channel, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # Shape: [batch_size, channel, 1]
        y = self.conv(y)  # Correct input shape for Conv1d
        y = self.bn(y)  # Shape: [batch_size, channel, 1]
        y = self.silu(y).transpose(-1, -2).unsqueeze(-1)  # Shape: [batch_size, channel, 1, 1]
        return x * y.expand_as(x)  # Element-wise multiplication


# CBAM+ECA
class Fusion(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(Fusion, self).__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention([3, 7])
        self.eca = ECA_layer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) 
        x = self.spatial_attention(x) 
        x = self.eca(x)
        return x


class ZSPA_NET(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, expansion=1, downsampling=False):
        super(ZSPA_NET, self).__init__()
    
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=s, padding=1, groups=c2, bias=False),  # Depthwise
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2 * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2 * self.expansion),
        )

        self.Fusion_block = Fusion(c2*self.expansion)

        if self.downsampling:
            self.downsampling = nn.Sequential(
                nn.Conv2d(c1, c2 * self.expansion, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(c2 * self.expansion)
            )
        
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.Fusion_block(out)

        if self.downsampling:
            residual = self.downsampling(x)
        
        out += residual
        out = self.activation(out)

        return out

    # def forward(self, x):
    #     print(f"Input shape: {x.shape}")
    #     residual = x
    #     out = self.bottleneck(x)
    #     print(f"After bottleneck: {out.shape}")
    #     out = self.Fusion_block(out)
    #     print(f"After Fusion_block: {out.shape}")

    #     if self.downsampling:
    #         residual = self.downsampling(x)
    #         print(f"After downsampling: {residual.shape}")

    #     out += residual
    #     out = self.activation(out)
    #     print(f"Output shape: {out.shape}")
    #     return out


# # Example usage
# x = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 input
# zpool = Zpool()
# output = zpool(x)
# print(output.shape)  # Should print (1, 6, 16, 16) if pooling halves the size

def test_ZSPA_NET():
    print("Testing ZSPA_NET and associated modules...\n")

    # Configuration
    batch_size = 2
    c1_channels = 16  # Input channels for ZSPA_NET
    c2_channels = 24  # Output channels for ZSPA_NET
    height, width = 32, 32  # Spatial dimensions
    kernel_size = 3
    stride = 2

    # Dummy input tensor
    x = torch.randn(batch_size, c1_channels, height, width)

    # 1. Test Zpool
    zpool = Zpool(kernel_size=kernel_size, stride=stride)
    zpool_out = zpool(x)
    print(f"Zpool output shape: {zpool_out.shape}\nExpected shape: "
          f"({batch_size}, {c1_channels * 2}, {height // stride}, {width // stride})")
    assert zpool_out.shape == (batch_size, c1_channels * 2, height // stride, width // stride), "Zpool failed!"

    # 2. ChannelAttention
    channel_attention = ChannelAttention(c1_channels)
    ca_out = channel_attention(x)
    print(f"ChannelAttention output shape: {ca_out.shape}\nExpected shape: {x.shape}")
    assert ca_out.shape == x.shape, "ChannelAttention failed!"

    # 3. SpatialAttention
    spatial_attention = SpatialAttention(kernel_sizes=[3, 7])
    sa_out = spatial_attention(x)
    print(f"SpatialAttention output shape: {sa_out.shape}\nExpected shape: {x.shape}")
    assert sa_out.shape == x.shape, "SpatialAttention failed!"

    # 4. ECA_layer
    eca = ECA_layer(c1_channels, k_size=kernel_size)
    eca_out = eca(x)
    print(f"ECA_layer output shape: {eca_out.shape}\nExpected shape: {x.shape}")
    assert eca_out.shape == x.shape, "ECA_layer failed!"

    # 5. Fusion
    fusion = Fusion(c1_channels)
    fusion_out = fusion(x)
    print(f"Fusion output shape: {fusion_out.shape}\nExpected shape: {x.shape}")
    assert fusion_out.shape == x.shape, "Fusion failed!"

    # 6. ZSPA_NET
    zspa_net = ZSPA_NET(c1_channels, c2_channels, k=1, s=1, expansion=2, downsampling=True)
    zspa_out = zspa_net(x)
    print(f"ZSPA_NET output shape: {zspa_out.shape}\nExpected shape: "
          f"({batch_size}, {c2_channels * 2}, {height}, {width})")
    assert zspa_out.shape == (batch_size, c2_channels * 2, height, width), "ZSPA_NET failed!"

    print("\nAll tests passed successfully!")


test_ZSPA_NET()


