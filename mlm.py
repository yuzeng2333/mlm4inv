import torch
import torch.nn as nn
import random
from dataloader import dataloader

# Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, dim_feedforward)
        transformer_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.decoder = nn.Linear(dim_feedforward, input_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.out_features)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


# Hyperparameters
input_size = 16  # Size of each data in the sequence
num_heads = 4    # Number of heads in the multi-head attention models
num_layers = 3   # Number of sub-encoder-layers in the encoder
dim_feedforward = 512  # Dimension of the feedforward network model in nn.TransformerEncoder
max_seq_len = 8  # Maximum length of the input sequence
num_samples = 1000

# Initialize the model
model = TransformerModel(input_size, num_heads, num_layers, dim_feedforward, max_seq_len)
dataloader = dataloader("./synthetic_many_vars/data/0.csv")

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch_idx, batch_data in enumerate(dataloader):
        # Masking a random element in each sequence of the batch
        mask_indices = np.random.randint(max_seq_len, size=batch_size)
        read_data = batch_data.clone()
        masked_data = batch_data.clone()
        for i, idx in enumerate(mask_indices):
            masked_data[i, idx, :] = 0

        # Forward pass
        output = model(masked_data)
        
        # Calculate loss only for the masked elements
        loss = sum(loss_function(output[i, idx, :], read_data[i, idx, :]) for i, idx in enumerate(mask_indices)) / batch_size

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

    # Add validation step here if you have a validation dataset

# Save the trained model
torch.save(model.state_dict(), "transformer_model.pth")
