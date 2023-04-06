import torch.optim
import scipy.sparse as sp
from utils import *
from function import *
from Params import *
import dill
import numpy as np
import matplotlib.pyplot as plt
from Model import Model
import torch.nn as nn

Test = True  # True, False


def main():
    version = 1378  # 6350
    output_file = '../data/output_{}/stage2/'.format(version)
    file = '../data/output_{}/stage1/'.format(version)
    data_path = file + 'records_patient.pkl'
    data = dill.load(open(data_path, 'rb'))  # 689/3175
    voc_path = output_file + 'voc_final.pkl'
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']
    med_dict = med_voc.idx2word
    # 标签1
    label_path = file + 'labels_med.pkl'
    labels = dill.load(open(label_path, 'rb'))  # (15016, 145)
    labels = torch.tensor(labels, device=args.device)
    # labels = torch.from_numpy(labels)

    label_d_path = file + 'labels_diag.pkl'
    labels_d = dill.load(open(label_d_path, 'rb'))  # (15016, 145)
    labels_d = torch.tensor(labels_d, device=args.device)
    # labels = torch.from_numpy(labels)

    # 同构路径
    ddi_adj_path = file + 'ddi_A_final.pkl'
    ehr_adj_path = file + 'ehr_adj_final.pkl'
    patient1_path = file + 'pmp_greater20.pkl'  # 大数据
    patient2_path = file + 'pdp_greater5.pkl'
    patient1 = dill.load(open(patient1_path, 'rb'))  # (51308545, 2),med,ndarry
    patient2 = dill.load(open(patient2_path, 'rb'))  # (112732620, 2)
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    # 异构路径
    pd_Hetergraph_path = file + 'pd_graph.pkl'
    pm_Hetergraph_path = file + 'pm_graph.pkl'
    pd_Hetergraph = dill.load(open(pd_Hetergraph_path, 'rb'))
    pm_Hetergraph = dill.load(open(pm_Hetergraph_path, 'rb'))
    args.patient = len(data)
    # 异构图
    pd_Hetertensor = makeTorchAdj(pd_Hetergraph, pd_Hetergraph.shape[1])
    pm_Hetertensor = makeTorchAdj(pm_Hetergraph, pm_Hetergraph.shape[1])  # [3287, 3287]:patient+med
    # 同构图
    ddi_adj_list = narry2list(ddi_adj)  # 改成邻接表
    ehr_adj_list = narry2list(ehr_adj)
    args.patient, args.med, args.diag = len(data), ddi_adj.shape[0], pd_Hetergraph.shape[1]
    # 把邻接矩阵处理成对称的
    patient1 = get_symmetrical_square_matrix(patient1, args.patient)
    patient2 = get_symmetrical_square_matrix(patient2, args.patient)
    pmp_graph = dgl.graph((patient1[:, 0], patient1[:, 1]), num_nodes=args.patient)
    pmp_graph = dgl.add_self_loop(pmp_graph)
    pdp_graph = dgl.graph((patient2[:, 0], patient2[:, 1]), num_nodes=args.patient)
    pdp_graph = dgl.add_self_loop(pdp_graph)
    ddi_graph = dgl.graph((ddi_adj_list[:, 0], ddi_adj_list[:, 1]), num_nodes=ddi_adj.shape[0])
    ehr_graph = dgl.graph((ehr_adj_list[:, 0], ehr_adj_list[:, 1]), num_nodes=ehr_adj.shape[0])
    g = [pmp_graph, pdp_graph, ddi_graph, ehr_graph]  # , diag_graph]

    # 数据集划分
    split_point = int(len(data) * 2 / 3)  #
    eval_len = int(len(data[split_point:]) / 2)  # 2503
    idx_train = torch.LongTensor(range(0, split_point))  # 10010
    idx_val = torch.LongTensor(range(split_point, split_point + eval_len))  # 2503
    idx_test = torch.LongTensor(range(split_point + eval_len, len(data)))  # 2503

    model = Model(
        num_meta_paths=len(g),
        patientnum=args.patient,  # 15016
        mednum=args.med,
        diagnum=args.diag,
        featuredim=args.featDim,  # 1870
        nhid=args.nhid,  # 8
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(args.device)
    g = [graph.to(args.device) for graph in g]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fcn = torch.nn.CrossEntropyLoss()
    if Test:
        resume_name = output_file + 'Test_Model_{}.model'.format(version)
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
        AUC, AP, top5, top10, top20 = evaluate(
            model, g, labels, idx_test, pm_Hetergraph, pm_Hetertensor, pd_Hetertensor, loss_fcn, plot=True,
            med_dict=med_dict
        )
        print(
            'Test AUC = %.4f, ' % AUC,
            'AP = %.4f,' % AP,
            'top5 = %.4f, ' % top5,
            'top10 = %.4f, ' % top10,
            'top20 = %.4f ' % top20,
        )
        return
    else:
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            model.train()
            output, diag, med, mEmbed_gcn, patient = model(pm_Hetertensor, pd_Hetertensor, g,
                                                           args.keepRate)  # torch.Size([1378, 136])
            loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()
            AUC, AP, top5, top10, top20 = evaluate(
                model, g, labels, idx_val, pm_Hetergraph, pm_Hetertensor, pd_Hetertensor, loss_fcn, plot=False
            )

            time = datetime.datetime.now()
            print('%s: |' % time,
                  'Epoch %d/%d, |' % (epoch, args.epochs),
                  'Train Loss=%.4f, ' % loss.item(),
                  'AUC=%.4f, ' % AUC,
                  'AP=%.4f |' % AP,
                  'top5=%.4f, ' % top5,
                  'top10=%.4f, ' % top10,
                  'top20=%.4f, ' % top20,
                  )
        torch.save(model.state_dict(), output_file + 'Test_Model_{}.model'.format(version))
        torch.save(model.state_dict(), output_file + 'Model.pth')
    # Test
    AUC, AP, top5, top10, top20 = evaluate(
        model, g, labels, idx_test, pm_Hetergraph, pm_Hetertensor, pd_Hetertensor, loss_fcn, plot=False
    )
    print(
        'Test AUC = %.4f, ' % AUC,
        'AP = %.4f,' % AP,
        'top5 = %.4f, ' % top5,
        'top10 = %.4f, ' % top10,
        'top20 = %.4f ' % top20,
    )


if __name__ == "__main__":
    main()
