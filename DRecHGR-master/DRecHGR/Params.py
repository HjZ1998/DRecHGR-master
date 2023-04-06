import argparse
import torch
import numpy as np

def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_heads', default=[8], type=list, help='num_heads')
    parser.add_argument('--nhid', default=8, type=int, help='')
    parser.add_argument('--featDim', default=64, type=int, help='')
    parser.add_argument('--hidden_units', default=8, type=int, help='')
    parser.add_argument('--dropout', default=0.6, type=float, help='dropout')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='dropout')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--patience', default=100, type=int, help='')

    parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')

    # RNN
    parser.add_argument('--emb_dim', default=64, type=int, help='RNN dimension')

    return parser.parse_args()

args = ParseArgs()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device:
    torch.cuda.manual_seed(args.seed)
