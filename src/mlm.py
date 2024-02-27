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
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file, num_epochs, MASK_IDX, data_file
from customAttn import CustomTransformerEncoderLayer, CustomTransformerEncoder
from transformer_train import transformer_train
from xgboost_train import xgboost_train
from dnn import dnn_train

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
    parser.add_argument('-e', '--epoch', default=num_epochs, type=int,
                        help="epoch number")
    parser.add_argument('-m', '--model', type=str, default='dnn',
                        help="choose the model type: transformer or xgboost")
    parser.add_argument('-l', '--load', action='store_true',
                        help="if true, load the model before training")
    return parser

def main(args):
    file_path = data_file
    #file_path = "../synthetic_many_vars/function_data.csv"
    if args.model == 'transformer':
        transformer_train(args, file_path)
    elif args.model == 'xgboost':
        xgboost_train(args, file_path)
    elif args.model == 'dnn':
        dnn_train(args, file_path)
    else:
        print("Invalid model type")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('masked language model for invariant generation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)