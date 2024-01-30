import torch

def sort_tensor(tensor, mask_idx):
  # assert the tensor is 3D
  assert tensor.dim() == 3
  values, indices = tensor[:, mask_idx, :].sort(dim=1)
  # get the second dimension
  seq_len = tensor.size(1)

  # Expand indices to use for gathering across the entire tensor
  indices_expanded = indices.unsqueeze(1).expand(-1, seq_len, -1)

  # Reorder the entire tensor based on the sorted indices of the first row
  tensor = torch.gather(tensor, 2, indices_expanded)
  return tensor