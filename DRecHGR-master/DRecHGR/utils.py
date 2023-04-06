import datetime
import errno
import os
import pickle
import random
import sys
from pprint import pprint
from Params import *
import numpy as np
import torch
from scipy import io as sio
from scipy import sparse
import scipy.sparse as sp
import dgl
from dgl.data.utils import _get_dgl_url, download, get_download_dir


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def makeTorchAdj(mat, itemNum):  # (15016, 145)
    # make ui adj
    a = sp.csr_matrix((args.patient, args.patient))  # (15016, 15016)
    b = sp.csr_matrix((itemNum, itemNum))  # (145,145)
    mat = sp.vstack([sp.hstack([a, mat]), sp.hstack(
        [mat.transpose(), b])])  # (15016, 15161), (145, 15161)vstack纵向拼接,hstack横向拼接,mat=(15161, 15161)
    mat = (mat != 0) * 1.0  #
    mat = (mat + sp.eye(mat.shape[0])) * 1.0  # 创建单位矩阵
    mat = normalizeAdj(mat)

    # make cuda tensor
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))  # torch.Size([2, 308005])
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(idxs, vals, shape).cuda()


def normalizeAdj(mat):
    degree = np.array(mat.sum(axis=-1))  # 对行求和
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])  #
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0  # 判断是否有无穷值
    dInvSqrtMat = sp.diags(dInvSqrt)  #
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def innerProduct(usrEmbeds, itmEmbeds):
    return torch.sum(usrEmbeds * itmEmbeds, dim=-1)


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
