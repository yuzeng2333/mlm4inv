import torch
import csv
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from util import sort_tensor
from config import MASK_IDX

def canonicalize(features_tensor, axis):
  # Calculate the mean and standard deviation for each column
  means = torch.mean(features_tensor, axis=axis, keepdim=True).detach()
  stds = torch.std(features_tensor, axis=axis, keepdim=True).detach()

  # Ensure that std is not zero to avoid division by zero
  stds[stds == 0] = 1
  
  # Subtract the mean and divide by the standard deviation for each column
  features_tensor = (features_tensor - means) / stds
  return features_tensor


def GenDataloader(file_path, batch_size, device, aug_data=False, shuffle=True):
  # Parameters
  
  # Initialize an empty list to hold the feature data
  features = []
  
  # Open the CSV file
  with open(file_path, newline='') as csvfile:
      reader = csv.reader(csvfile)
  
      # Skip the header row
      next(reader, None)
  
      # Read each row after the header
      for row in reader:
          cleaned_row = [float(cell.strip()) for cell in row]
          features.append(cleaned_row)

  features = np.array(features)
  
  # Convert numpy array to torch tensor
  features_tensor = torch.tensor(features).float()
  
  # Move the tensor to CUDA device
  features_tensor = features_tensor.to(device)

  # Convert to PyTorch tensors
  features_tensor = features_tensor.transpose(0, 1)
  rows = features_tensor.size(0)
  columns = features_tensor.size(1)
  if aug_data:
    features_tensor = canonicalize(features_tensor, 1)
    features_tensor = sort_tensor(features_tensor, MASK_IDX, size=2)
    smaller_tensor = torch.zeros((rows, columns), device=device)
    small_tensor = torch.zeros((rows, columns), device=device)
    large_tensor = torch.zeros((rows, columns), device=device)
    larger_tensor = torch.zeros((rows, columns), device=device)

    # Initialize idx tensors with the correct shape
    idx1 = torch.zeros((columns,), device=device, dtype=torch.int64)
    idx2 = torch.zeros((columns,), device=device, dtype=torch.int64)
    idx3 = torch.zeros((columns,), device=device, dtype=torch.int64)
    idx4 = torch.zeros((columns,), device=device, dtype=torch.int64)

    # generate the column indices
    for c in range(columns):
        if c == 0:
          # randomly pick two indices, idx1 <= idx2 <= c
          idx1[c] = 0
          idx2[c] = 0
        elif c == 1:
          idx1[c] = 0
          idx2[c] = 1
        else:
          idx1[c] = torch.randint(0, c-1, (1,), device=device)
          idx2[c] = torch.randint(idx1[c]+1, c, (1,), device=device)
        if c == columns - 2:
          idx3[c] = columns - 2
          idx4[c] = columns - 1
        elif c == columns - 1:
          idx3[c] = columns - 1
          idx4[c] = columns - 1
        else:
          idx3[c] = torch.randint(c+1, columns-1, (1,), device=device)
          idx4[c] = torch.randint(idx3[c]+1, columns, (1,), device=device)

    # Use the indices with torch.gather
    smaller_tensor = torch.gather(features_tensor, 1, idx1.unsqueeze(0).expand(rows, -1))
    small_tensor = torch.gather(features_tensor, 1, idx2.unsqueeze(0).expand(rows, -1))
    large_tensor = torch.gather(features_tensor, 1, idx3.unsqueeze(0).expand(rows, -1))
    larger_tensor = torch.gather(features_tensor, 1, idx4.unsqueeze(0).expand(rows, -1))

    smaller_tensor = smaller_tensor.unsqueeze(2)
    small_tensor = small_tensor.unsqueeze(2)
    large_tensor = large_tensor.unsqueeze(2)
    larger_tensor = larger_tensor.unsqueeze(2)
    features_tensor = features_tensor.unsqueeze(2)

    # Concatenate tensors along the desired dimension
    features_tensor = torch.cat((smaller_tensor, small_tensor, features_tensor, large_tensor, larger_tensor), 2)
    # shuffle along the second dimension
    shuffle_indices = torch.randperm(features_tensor.size(1), device=device)
    features_tensor = features_tensor[:, shuffle_indices, :]
  else:
    assert columns % 16 == 0
    #features_tensor = sort_tensor(features_tensor, MASK_IDX, 2)
    features_tensor = features_tensor.view(rows, int(columns / 16), 16)

  features_tensor = features_tensor.transpose(0, 1)
  # Create a TensorDataset
  dataset = TensorDataset(features_tensor)
  
  # Create a DataLoader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return dataloader 
