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
                                      dtype=np.float32)     # pmp:51308545/225480256(是1的)，占0.22755227; pdp:11617441/225480256,占0.0515231
    # 构建对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)     #
    # 加自环
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])        # m:102632106个1, d:23249898
    # 邻接矩阵转换成邻接表
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


def evaluate(model, g, labels, ids, pm_Hetergraph, pm_Hetertensor, pd_Hetertensor, loss_fcn, plot=False, med_dict=None):
    model.eval()
    with torch.no_grad():
        # patient, med = model(pm_Hetertensor, pd_Hetertensor, g, args.keepRate)  # torch.Size([15016, 145])
        output, diag, med, mEmbed_gcn, patient = model(pm_Hetertensor, pd_Hetertensor, g, args.keepRate)
    p_med_similarity = output[ids]
    # p_diag_similarity = output_d[ids]
    top5, top10, top20, top5locs, top10locs, top20locs = topkRecall(p_med_similarity, labels[ids])
    AUC = roc_auc_compute_fn(p_med_similarity, labels[ids])  # torch.Size([2503, 145])
    AP = get_ap(p_med_similarity, labels[ids])
    if plot:
        # pid = [9, 64,111, 117, 161,162,219,301,333, 365,449, 463, 480,483, 484, 485,510, 517]       # mimic数据集
        # pid = [2,4,5,10,18,19,20,29,32,35,36,52,53,57,58]       # demo数据集: [36,29,18]
        # pid = [36, 29, 18]      # 29
        # pid = [483, 484, 517]    # mimic3,483,517
        # pid = [298, 463, 485]  # 大数据集
        # pid = [19, 20, 99]  # 小数据集
        pid = [10, 36, 99]  # 小数据集2,
        # pid = [2, 5, 10, 19, 20, 35, 36, 53, 94, 99]
        for idx in pid:     # 展示出病人和药物embedding的相对位置,以及颜色表示是否预测出来
            p_idx = idx + ids[0].item()
            pEmbed = patient[p_idx].reshape(1, 64)
            # label med
            label = labels[p_idx].detach().cpu().numpy()      # label:numpy
            label = np.where([label == 1])[1]              # 标签id
            med_num = label.size
            mask = dict()       # 用于创建一个字典
            mask[p_idx] = "patient"
            # 预测id
            predict_set = set(top20locs[idx].detach().cpu().numpy().tolist())
            predict_true = set(label.tolist()) & predict_set    # 预测对的
            predict_false = set(label.tolist()) - predict_true  # 没预测出来的
            for false_id in predict_false:
                mask[false_id] = "Unpredicted drug"     # -1
            # 标签id
            for true_id in predict_true:
                mask[true_id] = "Predictive drug"       # 1
            # predict med
            idx_element_list = list(mask.keys())
            color_idx = {}
            for mid, flag in mask.items():
                color_idx.setdefault(flag, [])
                color_idx[flag].append(idx_element_list.index(mid))
            mEmbed = med[idx_element_list[1:]]
            embeddings = torch.cat([pEmbed, mEmbed], dim=0)     # 按列拼接
            embedding_array = embeddings.detach().cpu().numpy()
            tsne = TSNE(n_components=2)
            node_pos = tsne.fit_transform(embedding_array)
            ax = plt.subplot(1, 1, 1)  # 子图初始化
            color_map = {}
            color_map['patient'] = 'blue'
            color_map['Unpredicted drug'] = 'orange'
            color_map['Predictive drug'] = "green"
            size = 20
            for c, c_idx in color_idx.items():  # c是类, idx是节点的序号
                color = color_map.get(c)
                plt.scatter(node_pos[c_idx, 0], node_pos[c_idx, 1], s=200, c=color, label=c)  # 用于生成一个散点图
                # plt.setlabel()
            # 给每个点加标签,只给药物加,patient没必要
            # 获取点的对应编号med_dict
            for idx_m, med_id in enumerate(idx_element_list[1:]): # idx寻找位置，med_id寻找药物name
                # if med_id > 136 or med_id > 112:
                #     name = "patient"
                # else:
                idx_pos = idx_m + 1
                name = med_dict.get(med_id)
                ax.text(node_pos[idx_pos, 0]*1.01, node_pos[idx_pos, 1]*1.01, name,
                        fontsize=size, color="black", style="italic", weight="light",
                        verticalalignment='center', horizontalalignment='right', rotation=0)  # 给散点加标签
            plt.legend(bbox_to_anchor=(-0.1, 0.98), fontsize=size, loc='lower left', ncol=2, labelspacing=0.2,
                       handlelength=0.1)  # ncol： 图例的列的数量
            plt.xticks([])
            plt.yticks([])
            # plt.xlabel('{}'.format(idx), fontsize=14)
            plt.savefig('{}.pdf'.format(idx), bbox_inches='tight')
            plt.show()
    AUC = np.mean(AUC)
    AP = np.mean(AP)
    top5, top10, top20 = np.mean(top5), np.mean(top10), np.mean(top20)
    return AUC, AP, top5, top10, top20


def topkRecall(preb, labels):       # recall,预测的
    labels = labels.detach().cpu().numpy()
    tops_5 = []
    tops_10 = []
    tops_20 = []
    top5locs = dict()
    top10locs = dict()
    top20locs = dict()
    for i in range(preb.shape[0]):
        _, locs5 = preb[i].topk(k=5)
        _, locs10 = preb[i].topk(k=10)
        _, locs20 = preb[i].topk(k=20)
        label = np.where(labels[i] == 1)[0]
        top5 = len(set(locs5.cpu().numpy()) & set(label)) / len(set(label))
        top10 = len(set(locs10.cpu().numpy()) & set(label)) / len(set(label))
        top20 = len(set(locs20.cpu().numpy()) & set(label)) / len(set(label))
        tops_5.append(top5)
        tops_10.append(top10)
        tops_20.append(top20)
        #
        top5locs[i] = locs5
        top10locs[i] = locs10
        top20locs[i] = locs20
    # return np.mean(tops_5), np.mean(tops_10), np.mean(tops_20)
    return tops_5, tops_10, tops_20, top5locs, top10locs, top20locs


def roc_auc_compute_fn(pred_all, label_all):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    AUC = []
    for i in range(pred_all.shape[0]):
        y_true = label_all[i].cpu().numpy()
        y_pred = pred_all[i].cpu().detach().numpy()
        # metrics中的auc 需要输入已知求得的 fpr 与 tpr 值。
        # roc_auc_score则是不需要求这两个值，直接将预测概率值和label输入到函数中便可以求得auc值，省略了求这两个值的步骤，由函数本身代求。
        auc = roc_auc_score(y_true, y_pred, average='macro')
        AUC.append(auc)
    # return np.mean(AUC)
    return AUC


def calAUC(pred_all, label_all):
    AUC = []
    # AUC = 0
    label_all = label_all.cpu().numpy()
    pred_all = pred_all.cpu().numpy()
    # label_all = list(map(int, label_all))
    # label_all = list(map(int, label_all))
    for i in range(pred_all.shape[0]):      # 遍历每一个病人
        label = list(map(int, label_all[i]))
        pred = pred_all[i].tolist()
        posNum = len(list(filter(lambda s: s == 1, label)))
        negNum = len(label) - posNum
        sortedq = sorted(enumerate(pred), key=lambda x: x[1])  # 从小到大排序
        posRankSum = 0
        for j in range(len(pred)):
            if (label[j] == 1):
                posRankSum += list(map(lambda x: x[0], sortedq)).index(j) + 1  # 返回value=44所在的索引
        auc = (posRankSum - posNum * (posNum + 1) / 2) / (posNum * negNum)
        AUC.append(auc)
        # AUC += auc
    return np.mean(AUC)


def precision_auc(y_gt, y_prob):    # AP
    all_micro = []
    for b in range(len(y_gt)):
        all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
    return np.mean(all_micro)


def get_ap(pred_all, label_all):
    AP = []
    for i in range(pred_all.shape[0]):
        y_true = label_all[i].cpu().numpy()
        y_pred = pred_all[i].cpu().detach().numpy()
        ap = average_precision_score(y_true, y_pred)    # average_precision_score
        # AP = AP + ap
        AP.append(average_precision_score(y_true, y_pred))
    # return np.mean(AP)      # mAP
    return AP

