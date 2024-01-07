import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from Params import *
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import torch.nn.functional as F
from tqdm import tqdm
import dill
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties
import matplotlib as mpl


def narry2list(ndarry):
    choice = np.where(ndarry != 0)
    adj = list(map(list, zip(*choice)))
    adj = np.array(adj)#.astype(str)
    return adj


def get_symmetrical_square_matrix(mat, num_nodes):
    adj = sp.csr_matrix((np.ones(mat.shape[0]), (mat[:, 0], mat[:, 1])),
                                      shape=(num_nodes, num_nodes),
                                      dtype=np.float32)     
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)     
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])       
    choice = np.where(adj.A != 0)
    adj = list(map(list, zip(*choice)))
    adj = np.array(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
