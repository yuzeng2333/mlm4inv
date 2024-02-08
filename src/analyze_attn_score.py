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
from customAttn import CustomTransformerEncoderLayer, CustomTransformerEncoder
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, data_file, AUG_DATA, MASK_IDX
from transformer_train import TransformerModel
import matplotlib.pyplot as plt

# take the model file from input
#print("Give the model file path: ")
model_file = 'model_no_extra_token.pth'

# the purpose of the code here is to read the model parameters and do analysis
model = TransformerModel(input_size, num_heads, num_layers, dim_feedforward, max_seq_len)
model = torch.nn.DataParallel(model)
d_model = dim_feedforward

# Load the parameters from the file
model_parameters = torch.load(model_file)

# Load the parameters into the model
model.load_state_dict(model_parameters)
dataloader = GenDataloader(data_file, batch_size=32, device='cpu', aug_data=AUG_DATA, shuffle=False)
for batch_idx, batch_data in enumerate(dataloader):
    batch_data = batch_data[0]
    batch_indices = torch.arange(batch_data.size(0))
    masked_data = batch_data.clone()
    masked_data = torch.cat((masked_data[:, :MASK_IDX, :], masked_data[:, MASK_IDX+1:, :]), dim=1)
    ret = model(masked_data, ret_token=False, use_extra_token=False)
    attn_weight_list = ret['attn_weight_list']
    layer1_weights = attn_weight_list[0]
    layer2_weights = attn_weight_list[1]
    # the first dimension should be the same as the batch size
    assert layer1_weights.size(0) == 32
    assert layer2_weights.size(0) == 32
    # take average of the batch
    layer1_avg_weights = layer1_weights.mean(dim=0)
    # conver to numpy array
    layer1_avg_weights = layer1_avg_weights.detach().cpu().numpy()
    var_size = layer1_avg_weights.size(0)
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
    for i in range(5):
        axes[i].hist(layer1_avg_weights[i].numpy(), bins=10, alpha=0.75)
        axes[i].set_title(f'Histogram for layer1_avg_weights[{i}, :]')

    plt.tight_layout()
    plt.show()
    break
