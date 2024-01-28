import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Forward pass of the original TransformerEncoderLayer
        src2, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        if self.linear1 is not None:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        # Return both the output and the attention weights
        return src, attn_output_weights

class CustomTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__(encoder_layer, num_layers, norm)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weights_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_list.append(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        # Return the output and a list of attention weights from each layer
        return output, attn_weights_list

