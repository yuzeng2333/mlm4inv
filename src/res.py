import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))  # Element-wise addition (residual connection)

# Define the Autoencoder with Residual Blocks
class ResidualAutoencoder(nn.Module):
    def __init__(self):
        super(ResidualAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(7 * 16, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 256)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 7 * 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 7, 16)  # Reshape back to original size
        return x
