import torch
from mlm import TransformerModel
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file, MASK_IDX
from dataloader import GenDataloader, canonicalize
import torch.nn.functional as F
import numpy as np

# define the model
model = TransformerModel(input_size, num_heads, num_layers, dim_feedforward, max_seq_len)
model = torch.nn.DataParallel(model)
d_model = dim_feedforward

# Load the parameters from the file
model_parameters = torch.load(model_file)

# Load the parameters into the model
model.load_state_dict(model_parameters)

# Access multi-head attention layer
attention_layer = model.module.transformer_encoder.layers[0].self_attn
in_proj_weight = attention_layer.in_proj_weight

# Access the weights
q_weight = attention_layer.in_proj_weight[:d_model, :]
k_weight = attention_layer.in_proj_weight[d_model:2*d_model, :]
v_weight = attention_layer.in_proj_weight[2*d_model:, :]

# Access the biases
q_bias = attention_layer.in_proj_bias[:d_model]
k_bias = attention_layer.in_proj_bias[d_model:2*d_model]
v_bias = attention_layer.in_proj_bias[2*d_model:]

# load input to calculate the attention scores
batch_size = 32
dataloader = GenDataloader("../synthetic_many_vars/data/1.csv", batch_size, 'cpu', True)
compute_num = 1
idx = 0
attn_weight_list = []
for batch_idx, batch_data in enumerate(dataloader):
    idx += 1
    if idx > compute_num:
        break
    batch_data = batch_data[0]
    batch_data = canonicalize(batch_data, 2)
    batch_indices = torch.arange(batch_data.size(0))
    mask_indices = np.full(batch_size, MASK_IDX)
    mask_indices = torch.tensor(mask_indices)
    batch_data[batch_indices, mask_indices, :] = 0
    ret = model(batch_data, True)
    token = ret['token']
    attn_weight_list = ret['attn_weight_list']
    tokens = token[0, :, :]
    token_interest = tokens[MASK_IDX, :]

Q = torch.matmul(tokens, q_weight) + q_bias
K = torch.matmul(tokens, k_weight) + k_bias
V = torch.matmul(tokens, v_weight) + v_bias

# Calculate attention scores for the i-th token against all tokens (including itself)
# Q[i]: [qkv_size], K: [seq_len, qkv_size]
# convert token_interest to int
attention_scores = torch.matmul(Q[MASK_IDX], K.transpose(0, 1))  # [seq_len]

# Scaling by the square root of the dimension of K (assuming qkv_size = dimension of K)
d_k = K.size(-1)
scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k))

# Apply softmax to get the attention weights (a distribution over all tokens)
attention_weights = F.softmax(scaled_attention_scores, dim=-1)  # [seq_len]

# print the attention weights
print("== computed attention weights:")
print(attention_weights)
print("== attention weights from model:")
print(attn_weight_list[0][0, MASK_IDX, :])
