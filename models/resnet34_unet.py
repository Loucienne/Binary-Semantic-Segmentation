import torch
import torch.nn as nn

#This block implement the 2 blue arrows that we can see on the explainative schema
class Encoder(nn.Module):
    def __init__(self, nb_input_channel, nb_output_channel):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(nb_input_channel, nb_output_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(nb_output_channel))
        
    def forward(self, input):
        return self.operation(input)


class Decoder(nn.Module):
    def __init__(self, nb_input_channel, nb_output_channel):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(nb_input_channel, nb_output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_output_channel),
            nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True))
        self.cbam = CBAM(nb_output_channel)
        
    def forward(self, input):
        inter = self.operation(input)
        return self.cbam.forward(inter)

        
class RESNET_34_UNET(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.encode1 = Encoder(in_channel, 64)
        self.encode2 = Encoder(64, 64)
        self.encode3 = Encoder(64, 128)
        self.encode4 = Encoder(128, 256)
        self.encode5 = Encoder(256, 512)
        self.encode6 = Encoder(512, 256)
        self.decode1 = Decoder(768, 32)
        self.decode2 = Decoder(288, 32)
        self.decode3 = Decoder(160, 32)
        self.decode4 = Decoder(96, 32)
        self.decode5 = Decoder(32, 32)
        self.encode7 = Encoder(32, 1)

    def forward(self, input):
        inter = self.encode1(input)
        #inter = nn.MaxPool2d(kernel_size=2, stride=2)(inter)
        inter1 = self.encode2(inter)
        inter2 = self.encode3(inter1)
        inter3 = self.encode4(inter2)
        inter4 = self.encode5(inter3)
        inter = self.encode6(inter4)
        inter = torch.cat([inter, inter4], 1)
        inter = self.decode1(inter)
        inter = torch.cat([inter, inter3], 1)
        inter = self.decode2(inter)
        inter = torch.cat([inter, inter2], 1)
        inter = self.decode3(inter)
        inter = torch.cat([inter, inter1], 1)
        inter = self.decode4(inter)
        inter = self.decode5(inter)
        inter = self.encode7(inter)
        return inter
    












import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Avg Pool
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global Max Pool

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # Pass through FC layers
        max_out = self.fc(self.max_pool(x))  # Pass through FC layers
        out = avg_out + max_out  # Combine outputs
        return self.sigmoid(out) * x  # Apply attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise Avg Pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise Max Pooling
        out = torch.cat([avg_out, max_out], dim=1)  # Concatenate both
        return self.sigmoid(self.conv(out)) * x  # Apply attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)  # Apply Channel Attention
        x = self.spatial_attention(x)  # Apply Spatial Attention
        return x
