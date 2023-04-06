import torch
import torch.nn as nn
import numpy as np
import dill
from torch.optim import Adam
import torch.nn.functional as F
import datetime
from Model import Model
from Params import args
import dgl

init = nn.init.xavier_uniform_
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getMultiHotVec(seq_input, diseaseNum, idx):
    input = []
    for id, adm in enumerate(seq_input):
        disease = np.zeros((1, diseaseNum))  # multihot
        disease[:, adm[idx]] = 1
        disease = torch.tensor(disease, device=device, dtype=torch.float)#.reshape(-1)      # 1576
        input.append(disease)
    return input


class RNN(nn.Module):
    def __init__(self, emb_dim, voc_size, patientNum, model_path):
        super(RNN, self).__init__()
        model_path = model_path
        checkpoint = torch.load(model_path, map_location='cpu')
        self.representation_encoder = Model(
            num_meta_paths=4,
            patientnum=patientNum,  # (1958, 1430, 112)
            mednum=voc_size[2],
            diagnum=voc_size[0],
            featuredim=args.featDim,  # 1870
            nhid=args.nhid,  # 8
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
        self.voc_size = voc_size
        self.representation_encoder.load_state_dict(checkpoint)
        self.encoders = nn.GRU(emb_dim, emb_dim * 2, batch_first=True)

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, voc_size[2])
        )

    def forward(self, seq_input, adj1, adj2, gs, keepRate):
        input = getMultiHotVec(seq_input, self.voc_size[0], 0)
        _, diseaseEmbed, medEmbed, _, _ = self.representation_encoder(adj1, adj2, gs, keepRate)
        i1_seq = []
        for adm in input:       # adm=[[],[],[]]
            visitEmbed = torch.matmul(adm, diseaseEmbed)    # torch.Size([1, 64])
            visitEmbed = visitEmbed.unsqueeze(dim=0)        # torch.Size([1, 1, 64])
            i1_seq.append(visitEmbed)
        i1_seq = torch.cat(i1_seq, dim=1)   # [1, 1, 64]-[1, 2, 64]
        o1, h1 = self.encoders(  # torch.Size([1, 1, 128])
            i1_seq
        )
        patient_representations = o1.squeeze(dim=0)  # torch.Size([1, 128])
        queries = self.query(patient_representations)   # torch.Size([1, 64])
        query = queries[-1:]
        key_weights1 = F.softmax(torch.mm(query, medEmbed.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, medEmbed)
        output = self.output(torch.cat([query, fact1], dim=-1))     # torch.Size([1, 136])
        return output
