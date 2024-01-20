import torch
import csv
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import batch_size

def GenDataloader(file_path):
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
  
  # Assuming 'features' is your numpy array
  # Calculate the mean and standard deviation for each column
  means = np.mean(features, axis=0)
  stds = np.std(features, axis=0)
  
  # Ensure that std is not zero to avoid division by zero
  stds[stds == 0] = 1
  
  # Subtract the mean and divide by the standard deviation for each column
  features_norm = (features - means) / stds

  # Convert to PyTorch tensors
  features_tensor = torch.tensor(features_norm, dtype=torch.float32)
  features_tensor = features_tensor.transpose(0, 1)
  rows = features_tensor.size(0)
  columns = features_tensor.size(1)
  assert columns % 16 == 0
  features_tensor = features_tensor.view(rows, int(columns / 16), 16)
  features_tensor = features_tensor.transpose(0, 1)

  
  # Create a TensorDataset
  dataset = TensorDataset(features_tensor)
  
  # Create a DataLoader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader 
