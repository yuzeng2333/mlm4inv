import argparse
import torch
import torch.nn as nn
import random
import math
from dataloader import GenDataloader, canonicalize
import numpy as np
import torch.optim as optim
from res import ResidualAutoencoder
from util import sort_tensor
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file, num_epochs, MASK_IDX
from customAttn import CustomTransformerEncoderLayer, CustomTransformerEncoder


# Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, dim_feedforward)
        transformer_layer = CustomTransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, batch_first=True)
        self.transformer_encoder = CustomTransformerEncoder(transformer_layer, num_layers=num_layers)
        self.decoder = nn.Linear(dim_feedforward, input_size)
        self.trainable_mask_value = nn.Parameter(torch.randn((1, input_size)))

    def forward(self, src, ret_token = False):
        token = self.embedding(src)
        embed, attn_weight_list = self.transformer_encoder(token)
        embed = self.decoder(embed)
        #embed = self.decoder(token)
        #output = canonicalize(decoder_output, 2)
        if ret_token:
           return {'embed': embed, 'token': token, 'attn_weight_list': attn_weight_list}
        else:
          return {'embed': embed, 'attn_weight_list': attn_weight_list}
        #return decoder_output


def transformer_train(args, file_path):
  COMP_ALL = 0
  USE_TRANSFORMER = 1
  PRINT_PARAMS = 1
  USE_MASK = 1
  USE_RAND = 0
  AUG_DATA = 1
  if args.all:
    RUN_ALL_BATCH = 1
  else:
    RUN_ALL_BATCH = 0
  # check the visibility of the cuda
  print('cuda is available: ', torch.cuda.is_available())
  print('cuda device count: ', torch.cuda.device_count())
  
  
  # Initialize the model
  if USE_TRANSFORMER:
    model = TransformerModel(input_size, num_heads, num_layers, dim_feedforward, max_seq_len)
  else:  
    model = ResidualAutoencoder()
  model = torch.nn.DataParallel(model)
  device = args.device
  batch_size = args.batch_size
  model.to(device)
  dataloader = GenDataloader(file_path, batch_size, device, aug_data=AUG_DATA, shuffle=False)
  if AUG_DATA:
     criterion = nn.MSELoss(reduction='sum').to(device)
  else:
    criterion = nn.MSELoss(reduction='mean').to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate is 0.001 by default
  random_tensor = torch.randn((1, input_size))

  # Training Loop
  for epoch in range(args.epoch):
      model.train()  # Set the model to training mode
      total_loss = 0

      if RUN_ALL_BATCH:
        max_batch_idx = len(dataloader)
      else: 
        max_batch_idx = 96
      for batch_idx, batch_data in enumerate(dataloader):
          if batch_idx > max_batch_idx:
              break
          batch_data = batch_data[0]
          #batch_data = sort_tensor(batch_data, MASK_IDX)
          if not AUG_DATA:
            batch_data = canonicalize(batch_data, 2)
          # Masking a random element in each sequence of the batch
          if USE_RAND:
            mask_indices = np.random.randint(max_seq_len-1, size=batch_size)
          else:
            mask_indices = np.full(batch_size, MASK_IDX)
          mask_indices = torch.tensor(mask_indices)
          read_data = batch_data.clone()
          masked_data = batch_data.clone()

          # Create a tensor of batch indices
          batch_indices = torch.arange(masked_data.size(0)).to(device)

          # Mask the data
          # This will set masked_data[i, idx, :] to random values for each i and corresponding idx
          if USE_MASK:
            #masked_data[batch_indices, mask_indices, :] = random_tensor
            masked_data[batch_indices, mask_indices, :] = model.module.trainable_mask_value

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
          ret_token = False
          ret = model(masked_data, ret_token)
          output = ret['embed']
          attn_weight_list = ret['attn_weight_list']
          attn_weights = attn_weight_list[0]
         
          if COMP_ALL == 1:
            masked_output = output
            masked_data = read_data
          elif AUG_DATA:
            masked_output = output[batch_indices, mask_indices, 2]
            masked_data = read_data[batch_indices, mask_indices, 2]
          else:
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
      if epoch % 20 == 0:
        print(f"Epoch {epoch+1}/{args.epoch}, Average Loss: {avg_loss}")
  
      # Add validation step here if you have a validation dataset
      if epoch % 100 == 0:
        #if PRINT_PARAMS:
        #  for name, param in model.named_parameters():
        #    print(f"Name: {name}")
        # print attn weights
        print("attn_weights: ", attn_weights[0][1])
        print("attn_weights: ", attn_weights[1][1])
        print("attn_weights: ", attn_weights[2][1])
        # Save the trained model
        torch.save(model.state_dict(), model_file)
        random_index = np.random.randint(0, batch_size)
        print(f"Random Sample at Epoch {epoch+1}:")
        print("Masked Output:", masked_output[batch_indices].detach().cpu().numpy())
        print("Masked Data:", masked_data[batch_indices].cpu().numpy())