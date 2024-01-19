import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def dataloader(file_path):
  # Parameters
  batch_size = 32
  
  # Read the CSV file
  df = pd.read_csv(file_path)
  
  features = df.iloc[1:, :].values
  
  # Convert to PyTorch tensors
  features_tensor = torch.tensor(features, dtype=torch.float32)
  features_tensor = features_tensor.transpose(0, 1)
  rows = features_tensor.size(0)
  columns = features_tensor.size(1)
  assert columns % 16 == 0
  features_tensor = features_tensor.view(rows, columns / 16, 16)
  features_tensor = features_tensor.transpose(0, 1)

  
  # Create a TensorDataset
  dataset = TensorDataset(features_tensor)
  
  # Create a DataLoader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader 
