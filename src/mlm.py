import argparse
import torch
import torch.nn as nn
import random
import math
from dataloader import GenDataloader
from config import batch_size
import numpy as np
import torch.optim as optim

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('-p', '--print', action='store_true')
    return parser

# Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, dim_feedforward)
        transformer_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.decoder = nn.Linear(dim_feedforward, input_size)

    def forward(self, src):
        embed = self.embedding(src) * math.sqrt(self.embedding.out_features)
        encoder_output = self.transformer_encoder(embed)
        output = self.decoder(encoder_output)
        return output

def main(args):
  # check the visibility of the cuda
  print('cuda is available: ', torch.cuda.is_available())
  print('cuda device count: ', torch.cuda.device_count())
  
  # Hyperparameters
  input_size = 16  # Size of each data in the sequence
  num_heads = 4    # Number of heads in the multi-head attention models
  num_layers = 3   # Number of sub-encoder-layers in the encoder
  dim_feedforward = 512  # Dimension of the feedforward network model in nn.TransformerEncoder
  max_seq_len = 8  # Maximum length of the input sequence
  num_epochs = 3000
  model_file = "transformer_model.pth"
  
  # Initialize the model
  model = TransformerModel(input_size, num_heads, num_layers, dim_feedforward, max_seq_len)
  model = torch.nn.DataParallel(model)
  device = args.device
  model.to(device)
  dataloader = GenDataloader("../synthetic_many_vars/data/1.csv", device)
  criterion = nn.MSELoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.01)  # Learning rate is 0.001 by default
  

  # Training Loop
  for epoch in range(num_epochs):
      model.train()  # Set the model to training mode
      total_loss = 0
  
      for batch_idx, batch_data in enumerate(dataloader):
          batch_data = batch_data[0]
          # Masking a random element in each sequence of the batch
          mask_indices = np.random.randint(max_seq_len-1, size=batch_size)
          mask_indices = torch.tensor(mask_indices)
          read_data = batch_data.clone()
          masked_data = batch_data.clone()

          # Create a tensor of batch indices
          batch_indices = torch.arange(masked_data.size(0)).to(device)
          
          # Mask the data
          # This will set masked_data[i, idx, :] to 0 for each i and corresponding idx
          masked_data[batch_indices, mask_indices, :] = 0

          # print the predicted value with saved model parameters
          if args.print:
            state_dict = torch.load(model_file)
            model.load_state_dict(state_dict)
            model.eval()
            predict = model(masked_data)
            masked_pred = predict[0, mask_indices[0], :]
            masked_data = read_data[0, mask_indices[0], :]
            print("=== pred: ", masked_pred)
            print("=== target: ", masked_data)
            return 

          # Forward pass
          output = model(masked_data)
          
          masked_output = output[batch_indices, mask_indices, :]
          masked_data = read_data[batch_indices, mask_indices, :]

          # Calculate loss only for the masked elements
          #loss = sum(criterion(output[i, idx, :], read_data[i, idx, :]) for i, idx in enumerate(mask_indices)) / batch_size
          loss = criterion(masked_output, masked_data)
  
          # Backpropagation
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  
          total_loss += loss.item()
  
      avg_loss = total_loss / len(dataloader)
      print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
  
      # Add validation step here if you have a validation dataset
      if epoch % 100 == 0:
        # Save the trained model
        torch.save(model.state_dict(), model_file)
        random_index = np.random.randint(0, batch_size)
        print(f"Random Sample at Epoch {epoch+1}:")
        print("Masked Output:", masked_output[random_index].detach().cpu().numpy())
        print("Masked Data:", masked_data[random_index].cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('masked language model for invariant generation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
