import argparse
import torch
import torch.nn as nn
import random
import math
from dataloader import GenDataloader, canonicalize
import numpy as np
import torch.optim as optim
import xgboost as xgb
from res import ResidualAutoencoder
from util import sort_tensor
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file, num_epochs, MASK_IDX
from customAttn import CustomTransformerEncoderLayer, CustomTransformerEncoder
from transformer_train import transformer_train

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('-p', '--print', action='store_true')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help="batch size of the input data")
    parser.add_argument('-a', '--all', action='store_true',
                    help="run all batches")
    # add epoch number
    parser.add_argument('-e', '--epoch', default=1000, type=int,
                    help="epoch number")
    parser.add_argument('-m', '--model', type=str, 
                        help="choose the model type: transformer or xgboost")
    return parser



def main(args):
    if args.model == 'transformer':
        transformer_train(args)
    elif args.model == 'xgboost':
        pass
    else:
        print("Invalid model type")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('masked language model for invariant generation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
