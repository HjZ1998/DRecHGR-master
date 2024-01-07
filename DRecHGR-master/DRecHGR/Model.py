import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from Params import args
init = nn.init.xavier_uniform_


class Model(nn.Module):
    def __init__(self, num_meta_paths, patientnum, mednum, diagnum, featuredim, nhid, num_heads, dropout, device=torch.device('cpu:0')):
        super(Model, self).__init__()

        self.pEmbed = nn.Parameter(init(torch.empty(patientnum, featuredim)))
        self.mEmbed = nn.Parameter(init(torch.empty(mednum, featuredim)))
        self.dEmbed = nn.Parameter(init(torch.empty(diagnum, featuredim)))
        self.gcnLayer1 = GCNLayer()     
        self.gcnLayer2 = GCNLayer()     

        self.HANlayers = HANLayer(num_meta_paths, featuredim, nhid, num_heads[0], dropout)
        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(featuredim * 2, mednum)
        )
        self.med = mednum

    def forward(self, adj1, adj2, gs, keepRate):
        embeds1 = torch.concat([self.pEmbed, self.mEmbed], dim=0)
        embeds2 = torch.concat([self.pEmbed, self.dEmbed], dim=0)
        gnnLats1, gnnLats2 = [[] for _ in range(2)]  
        lats1, lats2 = [embeds1], [embeds2]
        for i in range(args.gnn_layer):     
            tem1 = self.gcnLayer1(adj1, lats1[-1])
            tem2 = self.gcnLayer2(adj2, lats2[-1])
            gnnLats1.append(tem1)
            gnnLats2.append(tem2)
        gnnEmbeds1 = sum(gnnLats1)    
        gnnEmbeds2 = sum(gnnLats2)    
        pEmbed_gcn1, pEmbed_gcn2 = gnnEmbeds1[:args.patient], gnnEmbeds2[:args.patient]     
        mEmbed_gcn = gnnEmbeds1[args.patient:]     
        dEmbed_gcn = gnnEmbeds2[args.patient:]     
        h = [pEmbed_gcn1, pEmbed_gcn2, mEmbed_gcn, mEmbed_gcn]#, dEmbed_gcn]
        patient, med = self.HANlayers(gs, h)
        patient1 = patient.unsqueeze(1)
        patient1 = patient1.repeat(1, self.med, 1)
        medication = med.unsqueeze(0).repeat(patient.shape[0], 1, 1)
        patient1 = patient1.reshape(-1, args.featDim)
        medication = medication.reshape(-1, args.featDim)
        simi_pm = torch.cat((patient1, medication), dim=1)   
        simi_pm = self.output1(simi_pm).reshape(-1, self.med, self.med)
        simi_pm = simi_pm.sum(axis=1)
        return simi_pm, dEmbed_gcn, med, mEmbed_gcn, patient


class HANLayer(nn.Module):
    def __init__(
        self, num_meta_paths, featuredim, nhid, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    featuredim,   
                    nhid,   # 8
                    layer_num_heads,   
                    dropout,
                    dropout,        
                    activation=F.elu,
                    allow_zero_in_degree=True
                ),
            )
        self.semantic_attention = SemanticAttention(  
            in_size=nhid * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):   
        embeddings = []
        for i, g in enumerate(gs):
            x = self.gat_layers[i](g, h[i])
            x = x.flatten(1)
            embeddings.append(x)   
        patient_semantic = []
        patient_semantic.append(embeddings[0])
        patient_semantic.append(embeddings[1])
        patient_semantic = torch.stack(
            patient_semantic, dim=1
        )
        med_semantic = []
        med_semantic.append(embeddings[2])
        med_semantic.append(embeddings[3])
        med_semantic = torch.stack(
            med_semantic, dim=1
        )
        # diag = embeddings[4]
        patient = self.semantic_attention(patient_semantic)
        med = self.semantic_attention(med_semantic)
        return patient, med


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.5)     

    def forward(self, adj, embeds):
        return self.act(torch.spmm(adj, embeds))


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):       
        w = self.project(z).mean(0)  
        beta = torch.softmax(w, dim=0)  
        beta = beta.expand((z.shape[0],) + beta.shape) 
        return (beta * z).sum(1)
