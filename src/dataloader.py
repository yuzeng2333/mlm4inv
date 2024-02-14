import torch
import csv
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from util import sort_tensor
from config import MASK_IDX
import matplotlib.pyplot as plt

def canonicalize_np(features_array, axis):
    # Calculate the mean and standard deviation for each column
    means = np.mean(features_array, axis=axis, keepdims=True)
    stds = np.std(features_array, axis=axis, keepdims=True)

    # Ensure that std is not zero to avoid division by zero
    stds[stds == 0] = 1
    
    # Subtract the mean and divide by the standard deviation for each column
    features_array = (features_array - means) / stds
    return features_array


def canonicalize(features_tensor, axis):
  # splot the 3rd row
  row_to_plot = features_tensor[2, :].cpu().numpy()

  # Plotting
  plt.plot(row_to_plot)
  plt.title("Plot of the 3rd Row of the Tensor")
  plt.xlabel("Column Index")
  plt.ylabel("Value")
  plt.show()
  # Calculate the mean and standard deviation for each column
  means = torch.mean(features_tensor, axis=axis, keepdim=True).detach()
  stds = torch.std(features_tensor, axis=axis, keepdim=True).detach()

  # Ensure that std is not zero to avoid division by zero
  stds[stds == 0] = 1
  
  # Subtract the mean and divide by the standard deviation for each column
  features_tensor = features_tensor - means
  features_tensor = features_tensor / stds
  row_to_plot = features_tensor[2, :].cpu().numpy()

  # Plotting
  plt.plot(row_to_plot)
  plt.title("Plot of the 3rd Row of the Tensor")
  plt.xlabel("Column Index")
  plt.ylabel("Value")
  plt.show()
  return features_tensor

def read_data(file_path):
  # Initialize an empty list to hold the feature data
  features = []
  # dict for variable names and their indices
  var_dict = {}
  
  # Open the CSV file
  with open(file_path, newline='') as csvfile:
      reader = csv.reader(csvfile)
      # read the first row
      header = next(reader)
      # fill the dict with variable names and their indices
      for i, var in enumerate(header):
          var_dict[var] = i 
  
      # Read each row after the header
      for row in reader:
          cleaned_row = [float(cell.strip()) for cell in row]
          features.append(cleaned_row)

  features = np.array(features)
  return features, var_dict

def split_tensor(tensor):
    # Create an empty list to hold the processed rows
    processed_rows = []
    
    for row in tensor:
        # Calculate max and min values of the current row
        max_val = torch.max(row)
        min_val = torch.min(row)
        
        # Check if the absolute value of max or min exceeds 10000
        if torch.abs(max_val) > 10000 or torch.abs(min_val) > 10000:
            raise ValueError("Absolute value of max or min exceeds 10000.")
        
        # Check if the absolute value of both max and min does not exceed 100
        if torch.abs(max_val) <= 100 and torch.abs(min_val) <= 100:
            # Do nothing and append the original row
            processed_rows.append(row.unsqueeze(0))
        else:
            # If the absolute value of either max or min is between 100 and 10000
            # Split the row into two: one with values % 100, and another with values / 100
            sign = torch.sign(row)
            row_abs = torch.abs(row)
            mod_row = row_abs % 100
            div_row = row_abs // 100
            mod_row = mod_row * sign
            div_row = div_row * sign
            processed_rows.append(mod_row.unsqueeze(0))
            processed_rows.append(div_row.unsqueeze(0))
    
    # Concatenate all processed rows into a new tensor
    new_tensor = torch.cat(processed_rows, dim=0)
    return new_tensor


def remove_outliers(data, n_std_dev=2):
    """
    Removes columns from the data tensor that contain outlier values in any row.
    Outliers are defined as values more than n_std_dev standard deviations from the mean of each row.
    
    Parameters:
    - data: A 2D PyTorch tensor where each row represents variables and each column represents a set of values.
    - n_std_dev: The number of standard deviations from the mean to consider a value as an outlier.
    
    Returns:
    - A 2D PyTorch tensor with outlier columns removed.
    """
    means = torch.mean(data, dim=1, keepdim=True)
    std_devs = torch.std(data, dim=1, keepdim=True)
    
    # Find the difference from the mean in units of standard deviation
    z_scores = torch.abs((data - means) / std_devs)
    
    # Identify columns that have any value more than n_std_dev standard deviations from the mean
    outlier_columns = torch.any(z_scores > n_std_dev, dim=0)
    
    # Invert to get columns to keep
    columns_to_keep = ~outlier_columns
    
    # Filter the data to keep only non-outlier columns
    filtered_data = data[:, columns_to_keep]
    
    return filtered_data


def GenDataloader(file_path, batch_size, device, aug_data=False, shuffle=True, random=False):
  features, var_dict = read_data(file_path)
  
  # Convert numpy array to torch tensor
  features_tensor = torch.tensor(features).float()
  
  # Move the tensor to CUDA device
  features_tensor = features_tensor.to(device)

  # Convert to PyTorch tensors
  features_tensor = features_tensor.transpose(0, 1)
  features_tensor = remove_outliers(features_tensor)
  #features_tensor = split_tensor(features_tensor)
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
          if random:
            idx1[c] = torch.randint(0, c-1, (1,), device=device)
            idx2[c] = torch.randint(idx1[c]+1, c, (1,), device=device)
          else:
            idx1[c] = c-2
            idx2[c] = c-1
        if c == columns - 2:
          idx3[c] = columns - 2
          idx4[c] = columns - 1
        elif c == columns - 1:
          idx3[c] = columns - 1
          idx4[c] = columns - 1
        else:
          if random:
            idx3[c] = torch.randint(c+1, columns-1, (1,), device=device)
            idx4[c] = torch.randint(idx3[c]+1, columns, (1,), device=device)
          else:
            idx3[c] = c+1
            idx4[c] = c+2

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
