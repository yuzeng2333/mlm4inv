import torch

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