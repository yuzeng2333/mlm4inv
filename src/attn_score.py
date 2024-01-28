import torch
from mlm import TransformerModel
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file

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

Q = torch.matmul(x, q_weight) + q_bias
K = torch.matmul(x, k_weight) + k_bias
V = torch.matmul(x, v_weight) + v_bias

# Calculate attention scores for the i-th token against all tokens (including itself)
# Q[i]: [qkv_size], K: [seq_len, qkv_size]
attention_scores = torch.matmul(Q[i], K.transpose(0, 1))  # [seq_len]

# Scaling by the square root of the dimension of K (assuming qkv_size = dimension of K)
d_k = K.size(-1)
scaled_attention_scores = attention_scores / torch.sqrt(d_k)

# Apply softmax to get the attention weights (a distribution over all tokens)
attention_weights = F.softmax(scaled_attention_scores, dim=-1)  # [seq_len]

# Attention output for the i-th token is a weighted sum of all value vectors
attention_output = torch.matmul(attention_weights, V)  # [qkv_size]