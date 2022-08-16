#utils.py
import torch
import torch.nn as nn
import torch.functional as F
import random

#processed_path = '//data//xuyuyang/BaiduIntern//processed_schnet//'
#processed_path = '/data2/xuyuyang/LIT-PCBA/processed_/'
processed_path = '/data2/xuyuyang/DEKOIS2.0/processed/'
random.seed(42)

class FocalLoss(nn.Module):
    def __init__(self,weight=None,gamma=2):
        super(FocalLoss, self).__init__()
        self.weight=weight
        self.gamma=gamma
    def set_weight(self, nweight):
        self.weight = nweight
    def forward(self, pred_y, targets):
        if self.weight == None:
            CE_loss=F.cross_entropy(pred_y, targets)
        else:
            CE_loss=F.cross_entropy(pred_y, targets, weight=self.weight)
        mask=targets.float()*(pred_y[:,0]**self.gamma)+(1-targets.float())*(pred_y[:,1]**self.gamma)
        return torch.mean(mask*CE_loss)

def load_ligand(file_path_num, pname, active, processed_path = processed_path):
    file_path = processed_path+'data_'+pname+'_'+active+'_'+str(file_path_num)+'.pt'
    data = torch.load(file_path)
    return data

def sample_datasets(data, task, n_way, m_support, n_query, processed_path = processed_path):
    actives, decoys=data
    support_list_a = random.sample(range(0,actives), m_support)
    support_list_d = random.sample(range(0,decoys), m_support)
    
    l_a = [i for i in range(0,actives) if i not in support_list_a]
    l_d = [i for i in range(0,decoys) if i not in support_list_d]
    query_list_a = random.sample(l_a, int(n_query/2))
    query_list_d = random.sample(l_d, n_query-int(n_query/2))
    
    support_dataset=[]
    query_dataset=[]
    for i in support_list_a:
        support_dataset.append(load_ligand(i, task, 'active', processed_path))
    for i in support_list_d:
        support_dataset.append(load_ligand(i, task, 'decoy', processed_path))
    random.shuffle(support_dataset)
    for i in query_list_a:
        query_dataset.append(load_ligand(i, task, 'active', processed_path))
    for i in query_list_d:
        query_dataset.append(load_ligand(i, task, 'decoy', processed_path))
    random.shuffle(query_dataset)
    
    return support_dataset, query_dataset

def sample_test_datasets(data, task, n_way, m_support, n_query, processed_path = processed_path):
    actives, decoys=data
    support_list_a = random.sample(range(0,actives), m_support)
    support_list_d = random.sample(range(0,decoys), m_support)
    
    l_a = [i for i in range(0,actives) if i not in support_list_a]
    l_d = [i for i in range(0,decoys) if i not in support_list_d]
    if len(l_a) <= int(n_query/2):
        query_list_a = l_a
    else:
        query_list_a = random.sample(l_a, int(n_query/2))
    if len(l_d) <= (n_query - int(n_query/2)):
        query_list_d = l_d
    else:
        query_list_d = random.sample(l_d, n_query-int(n_query/2))
    
    support_dataset=[]
    query_dataset=[]
    for i in support_list_a:
        support_dataset.append(load_ligand(i, task, 'active', processed_path))
    for i in support_list_d:
        support_dataset.append(load_ligand(i, task, 'decoy', processed_path))
    random.shuffle(support_dataset)
    for i in query_list_a:
        query_dataset.append(load_ligand(i, task, 'active', processed_path))
    for i in query_list_d:
        query_dataset.append(load_ligand(i, task, 'decoy', processed_path))
    random.shuffle(query_dataset)
    
    return support_dataset, query_dataset


