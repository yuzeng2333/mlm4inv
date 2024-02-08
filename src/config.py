batch_size = 32
input_size = 5 #16  # Size of each data in the sequence
num_heads = 8    # Number of heads in the multi-head attention models
num_layers = 2   # Number of sub-encoder-layers in the encoder
dim_feedforward = 64 #64  # Dimension of the feedforward network model in nn.TransformerEncoder
max_seq_len = 7  # Maximum length of the input sequence
num_epochs = 1000
MASK_IDX = 1
model_file = "transformer_model.pth"
MASK_VAR = 'y'
data_file = "../synthetic_many_vars/data/simple.csv"
AUG_DATA = 1