import torch
import torch.nn as nn
import torch.nn.functional as F
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file, num_epochs, MASK_IDX, AUG_DATA
from dataloader import GenDataloader, canonicalize
import torch.optim as optim
import numpy as np

# Define the model
class DivisionModel(nn.Module):
    def __init__(self, var_num, input_dim, hidden_dim, output_dim):
        super(DivisionModel, self).__init__()
        self.var_num = var_num 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define the first layer (input to first hidden layer)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Define the second, third, and fourth layers (hidden layers)
        self.fc2 = nn.Linear(self.var_num, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Define the final layer
        self.fc5 = nn.Linear(self.hidden_dim * self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        # Pass through the first layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        x = x.transpose(1, 2)

        # Pass through the second, third, and fourth layers with ReLU activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # flatten the 2nd and 3rd dimension of the tensor
        x = x.view(-1, self.hidden_dim * self.hidden_dim)
        # Pass through the final layer
        x = self.fc5(x)
        return x

def dnn_train(args, file_path):
  COMP_ALL = 0
  USE_TRANSFORMER = 1
  PRINT_PARAMS = 1
  USE_MASK = 1
  USE_RAND = 0
  USE_EXTRA_TOKEN = False
  if args.all:
    RUN_ALL_BATCH = 1
  else:
    RUN_ALL_BATCH = 0
  # check the visibility of the cuda
  print('cuda is available: ', torch.cuda.is_available())
  print('cuda device count: ', torch.cuda.device_count())
  
  var_num = 3
  input_dim = 5
  model = DivisionModel(var_num, input_dim=input_dim, hidden_dim=64, output_dim=1)
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

  if args.load:
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)

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
          masked_data = masked_data[:, MASK_IDX+1:MASK_IDX+4, :]
          #masked_data = torch.cat((masked_data[:, MASK_IDX+1:MASK_IDX+5, :], masked_data[:, MASK_IDX+5:MASK_IDX+6, :]), dim=1)
          #masked_data = torch.cat((masked_data[:, :MASK_IDX, :], masked_data[:, MASK_IDX+1:-2, :]), dim=1)
          #masked_data[batch_indices, 2, :] = 0
          assert var_num == masked_data.size(1)

          # create a src_key_padding_mask
          #src_key_padding_mask = torch.zeros((batch_size, 3)).to(device)
          #src_key_padding_mask[batch_indices, 2] = 0.9
          #src_key_padding_mask = src_key_padding_mask.bool()

          # Forward pass
          ret = model(masked_data)
          # flatten ret
          output = ret.squeeze()
          #attn_weight_list = ret['attn_weight_list']
          #attn_weights = attn_weight_list[0]
          #avg_batch_weights = attn_weights.mean(dim=0)
          #avg_attn_weights = avg_batch_weights.mean(dim=1)
         
          masked_data = read_data[batch_indices, mask_indices, 2]

          # Calculate loss only for the masked elements
          #loss = sum(criterion(output[i, idx, :], read_data[i, idx, :]) for i, idx in enumerate(mask_indices)) / batch_size
          loss = criterion(output, masked_data) #+ 0.005 * avg_attn_weights.sum()
  
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
      #  #if PRINT_PARAMS:
      #  #  for name, param in model.named_parameters():
      #  #    print(f"Name: {name}")
      #  # print attn weights
      #  print("attn_weights0: ", avg_batch_weights[0])
      #  #print("====================")
      #  #print("attn_weights1: ", attn_weights[0][1])
      #  #print("attn_weights1: ", attn_weights[1][1])
      #  #print("attn_weights1: ", attn_weights[2][1])
      #  #print("====================")
      #  #print("attn_weights2: ", attn_weights[0][2])
      #  #print("attn_weights2: ", attn_weights[1][2])
      #  #print("attn_weights2: ", attn_weights[2][2])
      #  # Save the trained model
      #  torch.save(model.state_dict(), model_file)
      #  random_index = np.random.randint(0, batch_size)
        print(f"Random Sample at Epoch {epoch+1}:")
        print("Masked Output:", output.detach().cpu().numpy())
        print("Masked Data:", masked_data.cpu().numpy())