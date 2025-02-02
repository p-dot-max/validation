
# class SelfAttention(nn.Module):
#     def __init__(self, c1, c2=None, *args, **kwargs):
#         super(SelfAttention, self).__init__()
#         self.in_channels = c1
#         self.out_channels = c2 if c2 is not None else c1  # Output channels default to input channels
#         self.query_conv = nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1)
#         self.key_conv   = nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
        
#     def forward(self, x):
#         B, C, H, W = x.size()
#         proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x H*W x C'
#         proj_key = self.key_conv(x).view(B, -1, H * W)                       # B x C' x H*W
#         energy = torch.bmm(proj_query, proj_key)                             # B x H*W x H*W
#         attention = F.softmax(energy, dim=-1)                                # B x H*W x H*W
#         proj_value = self.value_conv(x).view(B, -1, H * W)                   # B x C x H*W
    
#         out = torch.bmm(attention, proj_value.permute(0, 2, 1))              # B x H*W x C
#         out = out.permute(0, 2, 1).view(B, C, H, W)                          # B x C x H x W
#         out = self.gamma * out + x
#         return out

-------------------------------
NEW
--------------------------------




























## Using new methods-1
# class SpatialAttention(nn.Module):
#     """Spatial-attention module."""

#     def __init__(self, kernel_size=7):
#         """Initialize Spatial-attention module with kernel size argument."""
#         super().__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()

#     def forward(self, x):
#         """Apply channel and spatial attention on input for feature recalibration."""
#         return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

# class DepthwiseSeparableConv(nn.Module):
#     """
#     Depthwise separable convolution.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
   
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# class ResBlock_CBAM(nn.Module):
#     """
#     Residual Block with Convolutional Block Attention Module (CBAM) using Depthwise Separable Convolutions.
#     """

#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1,
#                  activation=True, expansion_factor=1, downsample=False):
#         super(ResBlock_CBAM, self).__init__()
#         self.expansion_factor = expansion_factor
#         self.downsample = downsample

#         # Define the bottleneck sequence with depthwise separable convolutions
#         self.bottleneck = nn.Sequential(
#             DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.1, inplace=True),
#             DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.1, inplace=True),
#             DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels * self.expansion_factor, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels * self.expansion_factor),
#         )

#         # Define the CBAM module
#         self.cbam = CBAM(in_channels=out_channels * self.expansion_factor)

#         # Define the downsampling layer if needed
#         if self.downsample:
#             self.downsample_layer = nn.Sequential(
#                 DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels * self.expansion_factor, kernel_size=1, stride=stride, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion_factor)
#             )

#         # Activation layer
#         self.activation_layer = nn.ReLU(inplace=True)

#     def forward(self, input_tensor):
#         """
#         Forward pass of the Residual Block with CBAM.
#         :param input_tensor: Input tensor of shape (batch_size, channels, height, width)
#         :return: Output tensor with CBAM applied
#         """
#         # Save the residual connection
#         residual_tensor = input_tensor

#         # Apply bottleneck layers
#         bottleneck_output = self.bottleneck(input_tensor)

#         # Apply CBAM
#         cbam_output = self.cbam(bottleneck_output)

#         # Downsample if necessary
#         if self.downsample:
#             residual_tensor = self.downsample_layer(input_tensor)

#         # Add residual connection
#         output_tensor = cbam_output + residual_tensor

#         # Apply activation
#         output_tensor = self.activation_layer(output_tensor)

#         return output_tensor

# class CBAM(nn.Module):
#     """
#     Convolutional Block Attention Module (CBAM).
#     """

#     def __init__(self, in_channels):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels)
#         # Add spatial attention if required
#         self.spatial_attention = SpatialAttention()

#     def forward(self, x):
#         x = self.channel_attention(x)
#         # x = self.spatial_attention(x)
#         return x

# class ChannelAttention(nn.Module):
#     """
#     Channel-attention module.
#     """

#     def __init__(self, in_channels: int):
#         super(ChannelAttention, self).__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.activation = nn.Sigmoid()

#     def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the channel attention module.
#         :param input_tensor: Input feature map of shape (batch_size, channels, height, width)
#         :return: Feature map with channel attention applied
#         """
#         pooled_tensor = self.global_avg_pool(input_tensor)
#         attention_scores = self.activation(self.fc(pooled_tensor))
#         return input_tensor * attention_scores


#Method 2 USED
# class DepthwiseSeparableConv(nn.Module):
#     """
#     Depthwise separable convolution layer.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
   
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# class ChannelAttention(nn.Module):
#     """
#     Channel-attention module.
#     """
#     def __init__(self, channels: int):
#         super(ChannelAttention, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = DepthwiseSeparableConv(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.act = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         pooled = self.pool(x)
#         return x * self.act(self.fc(pooled))

# class SpatialAttention(nn.Module):
#     """
#     Spatial-attention module.
#     """
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv = DepthwiseSeparableConv(2, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out = torch.max(x, dim=1, keepdim=True)[0]
#         return x * self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))

# class CBAM(nn.Module):
#     """
#     Convolutional Block Attention Module.
#     """
#     def __init__(self, c1, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(c1)
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         return self.spatial_attention(self.channel_attention(x))

# class ResBlock_CBAM(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, expansion=1, downsampling=False):
#         super(ResBlock_CBAM, self).__init__()
#         self.expansion = expansion
#         self.downsampling = downsampling

#         self.bottleneck = nn.Sequential(
#             DepthwiseSeparableConv(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.LeakyReLU(0.1, inplace=True),
#             DepthwiseSeparableConv(c2, c2, kernel_size=3, stride=s, padding=1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.LeakyReLU(0.1, inplace=True),
#             DepthwiseSeparableConv(c2, c2 * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(c2 * self.expansion),
#         )
       
#         self.cbam = CBAM(c1=c2 * self.expansion)

#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 DepthwiseSeparableConv(c1, c2 * self.expansion, kernel_size=1, stride=s, padding=0, bias=False),
#                 nn.BatchNorm2d(c2 * self.expansion)
#             )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.bottleneck(x)
#         out = self.cbam(out)
#         if self.downsampling:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out



--------------------------------------------------------------------

# class GAM_Attention(nn.Module):
#     def __init__(self, c1, c2, group=True, rate=4):
#         super(GAM_Attention, self).__init__()

#         self.channel_attention = nn.Sequential(
#             nn.Linear(c1, int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(c1 / rate), c1)
#         )

#         self.spatial_attention = nn.Sequential(

#             nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
#                                                                                                      kernel_size=7,
#                                                                                                      padding=3),
#             nn.BatchNorm2d(int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
#                                                                                                      kernel_size=7,
#                                                                                                      padding=3),
#             nn.BatchNorm2d(c2)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
#         x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
#         x_channel_att = x_att_permute.permute(0, 3, 1, 2)
#         # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
#         x = x * x_channel_att

#         x_spatial_att = self.spatial_attention(x).sigmoid()
#         x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
#         out = x * x_spatial_att
#         # out=channel_shuffle(out,4) #last shuffle
#         return out


# class MultiScaleGAM_Attention(nn.Module):
#     def __init__(self, c1, c2, group=True, rate=4):
#         super(MultiScaleGAM_Attention, self).__init__()
#         self.scales = [3, 5, 7]  # Example kernel sizes for multi-scale attention

#         self.channel_attention = nn.Sequential(
#             nn.Linear(c1, int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(c1 / rate), c1)
#         )

#         self.spatial_attentions = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(c1, c1 // rate, kernel_size=s, padding=s // 2, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
#                                                                                                                kernel_size=s,
#                                                                                                                padding=s // 2),
#                 nn.BatchNorm2d(int(c1 / rate)),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(c1 // rate, c2, kernel_size=s, padding=s // 2, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
#                                                                                                                kernel_size=s,
#                                                                                                                padding=s // 2),
#                 nn.BatchNorm2d(c2)
#             ) for s in self.scales
#         ])

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
#         x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
#         x_channel_att = x_att_permute.permute(0, 3, 1, 2)
#         x = x * x_channel_att

#         x_spatial_att_sum = 0
#         for sa in self.spatial_attentions:
#             x_spatial_att = sa(x).sigmoid()
#             x_spatial_att = channel_shuffle(x_spatial_att, 4)
#             x_spatial_att_sum += x_spatial_att

#         out = x * x_spatial_att_sum
#         return out


# def channel_shuffle(x, groups=2):
#     B, C, H, W = x.size()
#     out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
#     out = out.view(B, C, H, W)
#     return out


# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Squeeze
#         b, c, _, _ = x.size()
#         y = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
#         y = self.fc1(y)
#         y = F.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         # Excitation
#         return x * y

# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         b, c, h, w = x.size()
#         proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)  # B x N x C
#         proj_key = self.key_conv(x).view(b, -1, w * h)  # B x C x N
#         energy = torch.bmm(proj_query, proj_key)  # B x N x N
#         attention = self.softmax(energy)  # B x N x N
#         proj_value = self.value_conv(x).view(b, -1, w * h)  # B x C x N

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
#         out = out.view(b, c, h, w)
#         out = self.gamma * out + x
#         return out

# # TO be integrated
# class AESEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(AESEBlock, self).__init__()
#         self.se_block = SEBlock(in_channels, reduction)
#         self.self_attention = SelfAttention(in_channels)

#     def forward(self, x):
#         x_se = self.se_block(x)  # Apply SE block
#         x_sa = self.self_attention(x_se)  # Apply self-attention on the output of SE block
#         return x_sa

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Squeeze
#         b, c, _, _ = x.size()
#         y = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
#         y = self.fc1(y)
#         y = F.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         # Excitation
#         return x * y

# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# # TO be integrated
# class SSEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SSEBlock, self).__init__()
#         self.se_block = SEBlock(in_channels, reduction)
#         self.spatial_attention = SpatialAttention()

#     def forward(self, x):
#         x_se = self.se_block(x)  # Apply SE block
#         x_sa = self.spatial_attention(x_se)  # Apply spatial attention on SE block's output
#         return x_se * x_sa  # Combine spatial attention with SE block output

# New Attention novel

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for Channel Attention."""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Excitation
        return x * y

class SelfAttention(nn.Module):
    """Self-Attention Module to capture long-range dependencies."""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

class AESEBlock(nn.Module):
    """Attention-Enhanced SE Block with Self-Attention."""
    def __init__(self, in_channels, reduction=16):
        super(AESEBlock, self).__init__()
        self.se_block = SEBlock(in_channels, reduction)
        self.self_attention = SelfAttention(in_channels)

    def forward(self, x):
        x_se = self.se_block(x)
        x_sa = self.self_attention(x_se)
        return x_sa

class MultiScaleSpatialAttention(nn.Module):
    """Multi-Scale Spatial Attention Module."""
    def __init__(self, in_channels, reduction=16, kernel_sizes=[3, 5, 7], groups=1):
        super(MultiScaleSpatialAttention, self).__init__()
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=k, padding=k // 2, groups=groups),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, 1, kernel_size=k, padding=k // 2, groups=groups),
                nn.BatchNorm2d(1)
            ) for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_maps = []
        for sa in self.spatial_attentions:
            attention_map = sa(x)
            attention_maps.append(attention_map)
        # Combine attention maps
        combined_attention = torch.mean(torch.stack(attention_maps, dim=0), dim=0)
        attention = self.sigmoid(combined_attention)
        return x * attention

class UnderwaterAttentionModule(nn.Module):
    """Novel Attention Module for Underwater Object Detection."""
    def __init__(self, in_channels, reduction=16, kernel_sizes=[3, 5, 7], groups=1):
        super(UnderwaterAttentionModule, self).__init__()
        self.channel_attention = AESEBlock(in_channels, reduction)
        self.spatial_attention = MultiScaleSpatialAttention(in_channels, reduction, kernel_sizes, groups)
        self.self_attention = SelfAttention(in_channels)

    def forward(self, x):
        # Apply Channel Attention
        x_ca = self.channel_attention(x)
        # Apply Spatial Attention
        x_sa = self.spatial_attention(x_ca)
        # Combine Channel and Spatial Attention
        x_combined = x_ca + x_sa
        # Apply Self-Attention
        x_out = self.self_attention(x_combined)
        return x_out

-----------------------


# class GAM_Attention(nn.Module):
#     def __init__(self, c1, c2, group=True, rate=4):
#         super(GAM_Attention, self).__init__()

#         self.channel_attention = nn.Sequential(
#             nn.Linear(c1, int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(c1 / rate), c1)
#         )

#         self.spatial_attention = nn.Sequential(

#             nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
#                                                                                                      kernel_size=7,
#                                                                                                      padding=3),
#             nn.BatchNorm2d(int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
#                                                                                                      kernel_size=7,
#                                                                                                      padding=3),
#             nn.BatchNorm2d(c2)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
#         x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
#         x_channel_att = x_att_permute.permute(0, 3, 1, 2)
#         # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
#         x = x * x_channel_att

#         x_spatial_att = self.spatial_attention(x).sigmoid()
#         x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
#         out = x * x_spatial_att
#         # out=channel_shuffle(out,4) #last shuffle
#         return out


# class MultiScaleGAM_Attention(nn.Module):
#     def __init__(self, c1, c2, group=True, rate=4):
#         super(MultiScaleGAM_Attention, self).__init__()
#         self.scales = [3, 5, 7]  # Example kernel sizes for multi-scale attention

#         self.channel_attention = nn.Sequential(
#             nn.Linear(c1, int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(c1 / rate), c1)
#         )

#         self.spatial_attentions = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(c1, c1 // rate, kernel_size=s, padding=s // 2, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
#                                                                                                                kernel_size=s,
#                                                                                                                padding=s // 2),
#                 nn.BatchNorm2d(int(c1 / rate)),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(c1 // rate, c2, kernel_size=s, padding=s // 2, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
#                                                                                                                kernel_size=s,
#                                                                                                                padding=s // 2),
#                 nn.BatchNorm2d(c2)
#             ) for s in self.scales
#         ])

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
#         x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
#         x_channel_att = x_att_permute.permute(0, 3, 1, 2)
#         x = x * x_channel_att

#         x_spatial_att_sum = 0
#         for sa in self.spatial_attentions:
#             x_spatial_att = sa(x).sigmoid()
#             x_spatial_att = channel_shuffle(x_spatial_att, 4)
#             x_spatial_att_sum += x_spatial_att

#         out = x * x_spatial_att_sum
#         return out


# def channel_shuffle(x, groups=2):
#     B, C, H, W = x.size()
#     out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
#     out = out.view(B, C, H, W)
#     return out


# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Squeeze
#         b, c, _, _ = x.size()
#         y = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
#         y = self.fc1(y)
#         y = F.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         # Excitation
#         return x * y

# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         b, c, h, w = x.size()
#         proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)  # B x N x C
#         proj_key = self.key_conv(x).view(b, -1, w * h)  # B x C x N
#         energy = torch.bmm(proj_query, proj_key)  # B x N x N
#         attention = self.softmax(energy)  # B x N x N
#         proj_value = self.value_conv(x).view(b, -1, w * h)  # B x C x N

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
#         out = out.view(b, c, h, w)
#         out = self.gamma * out + x
#         return out

# # TO be integrated
# class AESEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(AESEBlock, self).__init__()
#         self.se_block = SEBlock(in_channels, reduction)
#         self.self_attention = SelfAttention(in_channels)

#     def forward(self, x):
#         x_se = self.se_block(x)  # Apply SE block
#         x_sa = self.self_attention(x_se)  # Apply self-attention on the output of SE block
#         return x_sa

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Squeeze
#         b, c, _, _ = x.size()
#         y = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
#         y = self.fc1(y)
#         y = F.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         # Excitation
#         return x * y

# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# # TO be integrated
# class SSEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SSEBlock, self).__init__()
#         self.se_block = SEBlock(in_channels, reduction)
#         self.spatial_attention = SpatialAttention()

#     def forward(self, x):
#         x_se = self.se_block(x)  # Apply SE block
#         x_sa = self.spatial_attention(x_se)  # Apply spatial attention on SE block's output
#         return x_se * x_sa  # Combine spatial attention with SE block output

# New Attention novel

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for Channel Attention."""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Excitation
        return x * y

class SelfAttention(nn.Module):
    """Self-Attention Module to capture long-range dependencies."""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

class AESEBlock(nn.Module):
    """Attention-Enhanced SE Block with Self-Attention."""
    def __init__(self, in_channels, reduction=16):
        super(AESEBlock, self).__init__()
        self.se_block = SEBlock(in_channels, reduction)
        self.self_attention = SelfAttention(in_channels)

    def forward(self, x):
        x_se = self.se_block(x)
        x_sa = self.self_attention(x_se)
        return x_sa

class MultiScaleSpatialAttention(nn.Module):
    """Multi-Scale Spatial Attention Module."""
    def __init__(self, in_channels, reduction=16, kernel_sizes=[3, 5, 7], groups=1):
        super(MultiScaleSpatialAttention, self).__init__()
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=k, padding=k // 2, groups=groups),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, 1, kernel_size=k, padding=k // 2, groups=groups),
                nn.BatchNorm2d(1)
            ) for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_maps = []
        for sa in self.spatial_attentions:
            attention_map = sa(x)
            attention_maps.append(attention_map)
        # Combine attention maps
        combined_attention = torch.mean(torch.stack(attention_maps, dim=0), dim=0)
        attention = self.sigmoid(combined_attention)
        return x * attention

class UnderwaterAttentionModule(nn.Module):
    """Novel Attention Module for Underwater Object Detection."""
    def __init__(self, in_channels, reduction=16, kernel_sizes=[3, 5, 7], groups=1):
        super(UnderwaterAttentionModule, self).__init__()
        self.channel_attention = AESEBlock(in_channels, reduction)
        self.spatial_attention = MultiScaleSpatialAttention(in_channels, reduction, kernel_sizes, groups)
        self.self_attention = SelfAttention(in_channels)

    def forward(self, x):
        # Apply Channel Attention
        x_ca = self.channel_attention(x)
        # Apply Spatial Attention
        x_sa = self.spatial_attention(x_ca)
        # Combine Channel and Spatial Attention
        x_combined = x_ca + x_sa
        # Apply Self-Attention
        x_out = self.self_attention(x_combined)
        return x_out