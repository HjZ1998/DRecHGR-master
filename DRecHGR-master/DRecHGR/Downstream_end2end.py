import torch
import torch.nn as nn
import numpy as np
import dill
from torch.optim import Adam
import torch.nn.functional as F
from function import topkRecall, roc_auc_compute_fn, get_ap, narry2list, get_symmetrical_square_matrix
import datetime
from Downstream_end2end_Model import RNN
from Params import args
from utils import EarlyStopping, makeTorchAdj
import dgl
from tqdm import tqdm
import os

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


def main():
    file = '/root/data/hjZ/DRecHGR-master/data/output_6350/stage'        # (1958, 1430, 112)
    model_path = file + '2/Model.pth'
    data_path = file + '2/records_final_stage2.pkl'   # 用于RNN的输入
    data = dill.load(open(data_path, 'rb'))     # 689
    patientNum = len(data)  # 3175
    args.patient = patientNum
    voc_path = file + '2/voc_final.pkl'
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    # stage1的输入
    ddi_adj_path = file + '1/ddi_A_final.pkl'
    ehr_adj_path = file + '1/ehr_adj_final.pkl'
    patient1_path = file + '2/pmp_greater29_finetune.pkl'
    patient2_path = file + '2/pdp_greater9_finetune.pkl'
    patient1 = dill.load(open(patient1_path, 'rb'))  # (51308545, 2),med,ndarry
    patient2 = dill.load(open(patient2_path, 'rb'))  # (112732620, 2)
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    pd_Hetergraph_path = file + '2/pd_graph_finetune.pkl'
    pm_Hetergraph_path = file + '2/pm_graph_finetune.pkl'
    pd_Hetergraph = dill.load(open(pd_Hetergraph_path, 'rb'))
    pm_Hetergraph = dill.load(open(pm_Hetergraph_path, 'rb'))
    ddi_adj_list = narry2list(ddi_adj)  # 改成邻接表
    ehr_adj_list = narry2list(ehr_adj)
    patient1 = get_symmetrical_square_matrix(patient1, args.patient)
    patient2 = get_symmetrical_square_matrix(patient2, args.patient)
    pmp_graph = dgl.graph((patient1[:, 0], patient1[:, 1]), num_nodes=args.patient)
    pmp_graph = dgl.add_self_loop(pmp_graph)
    pdp_graph = dgl.graph((patient2[:, 0], patient2[:, 1]), num_nodes=args.patient)
    pdp_graph = dgl.add_self_loop(pdp_graph)
    ddi_graph = dgl.graph((ddi_adj_list[:, 0], ddi_adj_list[:, 1]), num_nodes=ddi_adj.shape[0])
    ehr_graph = dgl.graph((ehr_adj_list[:, 0], ehr_adj_list[:, 1]), num_nodes=ehr_adj.shape[0])
    g = [pmp_graph, pdp_graph, ddi_graph, ehr_graph]
    pd_Hetertensor = makeTorchAdj(pd_Hetergraph, pd_Hetergraph.shape[1])
    pm_Hetertensor = makeTorchAdj(pm_Hetergraph, pm_Hetergraph.shape[1])

    # 划分数据集
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    LR = 1e-3
    EPOCH = 100     # 100
    Test = False    # True, False
    model = RNN(emb_dim=64, voc_size=voc_size, patientNum=patientNum, model_path=model_path)
    model.to(device=device)
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )
    g = [graph.to(args.device) for graph in g]
    stopper = EarlyStopping(patience=args.patience)
    optimizer = Adam(list(model.parameters()), lr=LR)
    time = datetime.datetime.now()
    print(time)
    if Test:
        resume_name = file + '2/Downstream{}.model'.format(len(data)*2)
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
        test_loss, AUC, AP, top5, top10, top20 = eval(model, data_test, voc_size, pm_Hetertensor, pd_Hetertensor, g, args.keepRate)
        print(
            'Test,'
            'AUC = %.4f, ' % AUC,
            'AP = %.4f, ' % AP,
            'top5 = %.4f, ' % top5,
            'top10 = %.4f, ' % top10,
            'top20 = %.4f, ' % top20,
        )
        print(len(data)*2)
        print(len(data_train))
    else:
        for epoch in range(EPOCH):
            model.train()
            loss_record = []
            for input in tqdm(data_train):  # input是每个病人的所有就诊序列
                for idx, adm in enumerate(input):
                    seq_input = input[:idx + 1]
                    loss1_target = np.zeros((1, voc_size[2]))  # 1*136
                    loss1_target[:, adm[2]] = 1
                    # disease = getMultiHotVec(seq_input, voc_size[0], 0)    # torch.Size([1, 1576])
                    # med_history = getMultiHotVec(seq_input, voc_size[2], 2)
                    output = model(seq_input, pm_Hetertensor, pd_Hetertensor, g, args.keepRate)     # torch.Size([1, 136])
                    loss = F.binary_cross_entropy_with_logits(output, torch.FloatTensor(loss1_target).to(device))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record.append(loss.item())
            val_loss, AUC, AP, top5, top10, top20 = eval(model, data_eval, voc_size, pm_Hetertensor, pd_Hetertensor, g, args.keepRate)
            early_stop = stopper.step(val_loss, AUC, model)
            time = datetime.datetime.now()
            # print(66)
            print('%s: |' % time,
                  'Epoch %d/%d, |' % (epoch, EPOCH),
                  'Train Loss=%.4f, ' % np.mean(loss_record),
                  'AUC=%.4f, ' % AUC,
                  'AP=%.4f |' % AP,
                  'top5=%.4f, ' % top5,
                  'top10=%.4f, ' % top10,
                  'top20=%.4f, ' % top20,
                  )
            if early_stop:
                break
        stopper.load_checkpoint(model)
        torch.save(model.state_dict(), open(os.path.join(file + '2/', 'Downstream{}.model'.format(len(data)*2)), 'wb'))
    # Test
    test_loss, AUC, AP, top5, top10, top20 = eval(model, data_test, voc_size, pm_Hetertensor, pd_Hetertensor, g, args.keepRate)
    print(
        'Test,'
        'AUC = %.4f, ' % AUC,
        'AP = %.4f, ' % AP,
        'top5 = %.4f, ' % top5,
        'top10 = %.4f, ' % top10,
        'top20 = %.4f, ' % top20,
    )


def eval(model, data_eval, voc_size, pm_Hetertensor, pd_Hetertensor, g, keepRate):
    # evaluate
    # print('')
    model.eval()
    AUCs = []
    APs = []
    tops_5 = []
    tops_10 = []
    tops_20 = []
    top5locs = []
    top10locs = []
    top20locs = []
    loss_record = []
    for step, input in enumerate(data_eval):  # input是每个病人的所有就诊序列
        for idx, adm in enumerate(input):
            seq_input = input[:idx + 1]
            loss1_target = np.zeros((1, voc_size[2]))  # 1*136
            loss1_target[:, adm[2]] = 1
            loss1_target = torch.FloatTensor(loss1_target).to(device)
            output = model(seq_input, pm_Hetertensor, pd_Hetertensor, g, args.keepRate)
            loss = F.binary_cross_entropy_with_logits(output, loss1_target)
            top5, top10, top20, top5loc, top10loc, top20loc = topkRecall(output, loss1_target)
            AUC = roc_auc_compute_fn(output, loss1_target)
            AP = get_ap(output, loss1_target)
            AUCs.append(AUC)
            APs.append(AP)
            tops_5.append(top5)
            tops_10.append(top10)
            tops_20.append(top20)
            top5locs.append(top5loc)
            top10locs.append(top10loc)
            top20locs.append(top20loc)
            loss_record.append(loss.item())
    return np.mean(loss_record), np.mean(AUCs), np.mean(APs), np.mean(tops_5), np.mean(tops_10), np.mean(tops_20)


if __name__ == "__main__":
    main()
