import torch 
import torch.nn as nn

class MultiScalePatchTranslator(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        # Each CNN handles a different patch size
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        # Final fusion layer
        self.final = nn.Conv2d(16 * 3, num_channels, kernel_size=1)

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        # Concatenate along channel dimension
        combined = torch.cat([out3, out5, out7], dim=1)
        return self.final(combined)

class PatchTransformer(nn.Module):
    def __init__(self, num_colors, num_patches, embed_dim, num_heads, num_layers):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.to_output = nn.Linear(embed_dim, num_colors)

    def forward(self, x):  # x shape: (B, num_patches, embed_dim)
        x = x + self.positional_encoding
        x = self.transformer(x)
        return self.to_output(x.mean(dim=1))  # global average or CLS token