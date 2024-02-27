import torch
import os
import re

def sort_tensor(tensor, mask_idx, size=3):
  # assert the tensor is 3D
  assert tensor.dim() == size
  if size == 3:
    values, indices = tensor[:, mask_idx, :].sort(dim=1)
    # get the second dimension
    seq_len = tensor.size(1)

    # Expand indices to use for gathering across the entire tensor
    indices_expanded = indices.unsqueeze(1).expand(-1, seq_len, -1)

    # Reorder the entire tensor based on the sorted indices of the first row
    tensor = torch.gather(tensor, 2, indices_expanded)
    return tensor
  elif size == 2:
    values, indices = tensor[mask_idx, :].sort()
    # get the second dimension
    seq_len = tensor.size(0)

    # Expand indices to use for gathering across the entire tensor
    indices_expanded = indices.unsqueeze(0).expand(seq_len, -1)

    # Reorder the entire tensor based on the sorted indices of the first row
    tensor = torch.gather(tensor, 1, indices_expanded)
    return tensor


def find_largest_number_in_csv_files(directory):
    # Regular expression pattern to match files named with a number followed by .csv
    pattern = re.compile(r'^(\d+)\.csv$')
    
    max_number = None
    
    # List all files in the given directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        match = pattern.match(filename)
        if match:
            # Extract the number from the filename and convert it to an integer
            number = int(match.group(1))
            # Update max_number if this number is larger than the current max_number
            if max_number is None or number > max_number:
                max_number = number
                
    return max_number