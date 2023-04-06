import dill
import numpy as np
import scipy.sparse as sp
import pickle
import torch

version = 6350      # 1378
input_file = 'output_{}/'.format(version)
output_file = input_file + 'stage'

ddi_adj_path = input_file + 'stage1/ddi_A_final.pkl'
ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
num_ddi_pair = np.where(ddi_adj == 1)[0]    # 876/674

data_path = input_file + 'records_final.pkl'
all_data = dill.load(open(data_path, 'rb'))     # 1378
# 划分stage1和stage2的数据
split_point = int(len(all_data) / 2)    # 689
data_stage1 = all_data[:split_point]
data_stage2 = all_data[split_point:]
dill.dump(data_stage2, open(output_file + '2/' + 'records_final_stage2.pkl', 'wb'))
print(len(data_stage1))
EHR_records = []
for pat in data_stage1:
    diag = []
    pro = []
    med = []
    for visit in pat:
        diag_set = set(diag) | set(visit[0])        #
        pro_set = set(pro) | set(visit[1])
        meg_set = set(med) | set(visit[2])
        diag = list(diag_set)
        pro = list(pro_set)
        med = list(meg_set)
    EHR_records.append([diag, pro, med])
dill.dump(EHR_records, open(output_file + '1/' + 'records_patient.pkl', 'wb'))
print('get records!')
# 生成异构图

voc_file = input_file + 'voc_final.pkl'
voc = dill.load(open(voc_file, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
patientNum = len(EHR_records)     # 3175-6998次就诊
diagNum = len(diag_voc.idx2word)  # 1958
medNum = len(med_voc.idx2word)    # 112
diag_p = []
med_p = []
for idx, visit in enumerate(EHR_records):
    med = visit[2]
    diag = visit[0]
    for med_i in med:
        med_p.append([idx, med_i])
    for diag_i in diag:
        diag_p.append([idx, diag_i])
# list转换成矩阵
diag_p = np.array(diag_p)      # (8091, 2) (75300, 2)
med_p = np.array(med_p)        # (12376, 2) (87650, 2)
pd_graph = sp.csr_matrix((np.ones(diag_p.shape[0]), (diag_p[:, 0], diag_p[:, 1])),
                                shape=(patientNum, diagNum),
                                dtype=np.float32)       # (267, 1097)
pm_graph = sp.csr_matrix((np.ones(med_p.shape[0]), (med_p[:, 0], med_p[:, 1])),
                                shape=(patientNum, medNum),     # ValueError:列索引超过矩阵维度
                                dtype=np.float32)       # (267, 126)
pm_graph = sp.coo_matrix(pm_graph)
pd_graph = sp.coo_matrix(pd_graph)

pickle.dump(pm_graph, open(output_file + '1/' + 'pm_graph.pkl', 'wb'))
pickle.dump(pd_graph, open(output_file + '1/' + 'pd_graph.pkl', 'wb'))
print('get hypergraph!')
# 生成同构图
patient1 = np.zeros((patientNum, patientNum))       # disease
patient2 = np.zeros((patientNum, patientNum))       # diagnose
patient_med_list = []
patient_diag_list = []
for patient in EHR_records:
    patient_med_list.append(patient[2])     # 得到每个人吃过的药的列表
    patient_diag_list.append(patient[0])

if version == 1378:
    med_gaps = [20]
    diag_gaps = [5]
else:
    med_gaps = [29]
    diag_gaps = [9]


def printPmpgraph(med):
    # patient-med-patient
    list1 = []
    for i, med_i in enumerate(patient_med_list):  # i指病人序号，med_i指病人对应的药物组合
        for j, med_j in enumerate(patient_med_list):
            if j <= i:
                continue
            if len(set(med_i) & set(med_j)) > med:
                list1.append([i, j])
    list1 = np.array(list1)  # (19916, 2)
    dill.dump(list1, open(output_file + '1/' + 'pmp_greater{}.pkl'.format(med), 'wb'))
    print("get med array{}".format(med))
    return list1


def printPdpgraph(diag):
    # patient-diag-patient
    list2 = []
    for i, diag_i in enumerate(patient_diag_list):
        for j, diag_j in enumerate(patient_diag_list):
            if j <= i:
                continue
            if len(set(diag_i) & set(diag_j)) > diag:
                list2.append([i, j])
    list2 = np.array(list2)
    dill.dump(list2, open(output_file + '1/' + 'pdp_greater{}.pkl'.format(diag), 'wb'))
    print("get diag array{}".format(diag))
    return list2

for med in med_gaps:
    list1 = printPmpgraph(med)

for diag in diag_gaps:
    list2 = printPdpgraph(diag)

labels = []
for id, adm in enumerate(EHR_records):
    target = np.zeros((1, medNum))
    target[:, adm[2]] = 1
    # target_d = np0]] = 1
    labels.append(target)  # loss1_target[0]
labels_m = np.array(labels).reshape(len(EHR_records), -1)      # (1763, 136)
dill.dump(labels_m, open(output_file + '1/' + 'labels_med.pkl', 'wb'))
