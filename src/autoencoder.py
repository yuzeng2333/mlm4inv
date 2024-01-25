class Autoencoder(nn.Module):
    def __init__(self, input_size, input_len):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.input_len = input_len
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size * input_len, 128),  # Flatten and reduce to 128
            nn.ReLU(),
            nn.Linear(128, 128),            
            nn.ReLU(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 128) 
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, input_size * input_len),  # Decompress back to original size
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), self.input_len, self.input_size)  # Reshape back to original size
        return x

