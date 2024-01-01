import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_Attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        # normalization helps prevent oscillations in the loss and therefore
        # helps reduce training time (number of epochs)
        
        # group norm is like a layer-norm but split up
        # So instead of the normalization effecting all features for each item/image
        # the normalization is for some features for each item/image
        # number of group is first argument, so 32 in this case
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, H, W)
        residue = x
        n,c,h,w = x.shape
        
        # (batch_size, features, H, W) -> (batch_size, features, H * W)
        x = x.view(n,c,h*w)
        
        # (batch_size, features, H*W) -> (batch_size, H * W, features)
        # kind of like attention LLM where we have (batch_size, seq_len, dim)
        x = x.transpose(-1,-2)
        
        # (batch_size, H * W, features) -> (batch_size, H * W, features)
        x = self.attention(x)
        
        # (batch_size, H * W, features) -> (batch_size, features, H*W)
        x = x.transpose(-1, -2)
        
        # (batch_size, features, H * W) -> (batch_size, features, H, W)
        x = x.view(n,c,h,w)
        
        x += residue
        
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 == nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 == nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1)
        
        
        # for skip connection
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, inC, H, W) -> (batch_size, outC, H, W)        
        residue = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)
    
    
        

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, 4, Height/8, Width/8) -> (batch_size, 4, Height/8, Width/8)
            nn.Conv2d(4,4,kernel_size=1, padding=0),
            
             # (batch_size, 4, Height/8, Width/8) -> (batch_size, 512, Height/8, Width/8)
            nn.Conv2d(4,512,kernel_size=3, padding=1),
            
            # (batch_size, 512, H/8, W/8) - >(batch_size, 512,H/8,W/8)
            VAE_ResidualBlock(512,512),
            
            # (batch_size, 512, H/8, W/8) - >(batch_size, 512,H/8,W/8)
            VAE_Attention(512),
            
            # (batch_size, 512, H/8, W/8) - >(batch_size, 512,H/8,W/8)
            VAE_ResidualBlock(512,512),

            # (batch_size, 512, H/8, W/8) - >(batch_size, 512,H/8,W/8)
            VAE_ResidualBlock(512,512),
            
            # (batch_size, 512, H/8, W/8) - >(batch_size, 512,H/8,W/8)
            VAE_ResidualBlock(512,512),
            
           
            
            # (batch_size, 512,H/8,W/8) -> (batch_size, 512,H/4,W/4)
            # replicate pixels down and right by default
            nn.Upsample(scale_factor=2),
            
            # (batch_size, 512, Height/4, Width/4) -> (batch_size, 512, Height/4, Width/4)
            nn.Conv2d(512,512,kernel_size=3, padding=1),
            
            # (batch_size, 512, Height/4, Width/4) -> (batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # (batch_size, 512,H/4,W/4) -> (batch_size, 512,H/2,W/2)
            nn.Upsample(scale_factor=2),
            
            # (batch_size, 512, Height/2, Width/2) -> (batch_size, 512, Height/2, Width/2)
            nn.Conv2d(512,512,kernel_size=3, padding=1),
            
            # (batch_size, 512, Height/2, Width/2) -> (batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            # (batch_size, 256,H/2,W/2) -> (batch_size, 256,H,W)
            nn.Upsample(scale_factor=2),
            
            # (batch_size, 256, Height, Width) -> (batch_size, 256, Height, Width)
            nn.Conv2d(256,256,kernel_size=3, padding=1),
            
            # (batch_size, 256, Height, Width) -> (batch_size, 128, Height, Width)
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            
            nn.GroupNorm(32,128),
            nn.SiLU(),
            
            # (batch_size, 128, Height, Width) -> (batch_size, 3, Height, Width)
            nn.Conv2d(128,3,kernel_size=3, padding=1)
            
            
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, H/8, W/8)
        
        # scale the input by a constant
        x /= 0.18215
            
        for module in self:              
            x = module(x)
        
        # (batch_size, 3, Height, Width)
        return x