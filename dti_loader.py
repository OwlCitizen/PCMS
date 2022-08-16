#dti_loader.py
import time
import torch
import pandas as pd
import random
#from concurrent.futures.thread import ThreadPoolExecutor
#from concurrent.futures._base import as_completed

#data_path='//data//xuyuyang//BaiduIntern//all//'
#processed_path = '//data//xuyuyang/BaiduIntern//processed_schnet//'
random.seed(42)

class ProteinDataset():
    def __init__(self, pname, len_act, len_dcy, pfile, fasta, processed_path):
        #print('loading '+pname+' ...')
        ##active_list=Chem.SDMolSupplier(data_path+'//'+pname+'//actives_final.sdf')
        ##decoy_list=Chem.SDMolSupplier(data_path+'//'+pname+'//decoys_final.sdf')
        self.len_active = len_act
        self.len_decoy = len_dcy
        ##if os.path.exists(processed_path+pname+'_processed.csv'):
        #info=pd.read_csv(processed_path+pname+'_processed.csv',index_col = 'Unnamed: 0')
        #self.pinfo = info
        #self.len_active = info.len_active.values[0]
        #self.len_decoy = info.len_decoy.values[0]
        self.processed_path = processed_path
        self.protein_file = pfile
        self.fasta = fasta
    def get_ligands(self):
        return self.len_active, self.len_decoy
    def get_protein(self):
        return (load_protein(self.protein_file), self.fasta)
                
def load_protein(file_name):
    protein = torch.load(file_name)
    return protein

def load_dataset(task, len_act, len_dcy, pfile, fasta, processed_path):#, processed_path = processed_path):
    dataset = ProteinDataset(task, len_act, len_dcy, pfile, fasta, processed_path)
    return (task, dataset)

def load_dataset_lp(task, pname, len_act, len_dcy, pfile, fasta, processed_path):#, processed_path = processed_path):
    dataset = ProteinDataset(task, len_act, len_dcy, pfile, fasta, processed_path)
    return (task, dataset, pname)

def load_dti_data(num_train_tasks, used_train_task, processed_path):
    start = time.time()
    data_info = pd.read_csv(processed_path+'data_info.csv',index_col='Unnamed: 0')
    train_tasks=list(data_info.loc[data_info.train==1].index)
    eval_tasks=list(data_info.loc[data_info.train==0].index)
    test_tasks=list(data_info.loc[data_info.train==-1].index)
    
    num_test_tasks = len(test_tasks)
    assert num_train_tasks <= len(train_tasks)
    assert used_train_task <= num_train_tasks
    #train_tasks=random.sample(train_tasks, num_train_tasks)
    train_tasks = train_tasks[:used_train_task]
    random.shuffle(train_tasks)
    
    train_datasets=[]
    test_datasets=[]
    eval_datasets = []
    #executor = ThreadPoolExecutor(max_workers=5)
    #all_task=[]
    for tname in train_tasks:
        #print(tname,'thread start')
        #if os.path.exists('//data//xuyuyang//BaiduIntern//processed//processed_dataset//'+tname+'_processeddataset.pt'):
        #    dataset = torch.load('//data//xuyuyang//BaiduIntern//processed//processed_dataset//'+tname+'_processeddataset.pt')
        #    train_datasets.append((tname,dataset))
        #else:
        #task = executor.submit(load_dataset,tname,all_pdb,processed_path)
        #all_task.append(task)
    #for future in as_completed(all_task):
        #(tname, dataset) = future.result()
        #torch.save(dataset, '//data//xuyuyang//BaiduIntern//processed//processed_dataset//'+tname+'_processeddataset.pt')
        train_datasets.append(load_dataset(tname, data_info.loc[tname].len_active, \
                                           data_info.loc[tname].len_decoy, \
                                               data_info.loc[tname].new_protein_filename, \
                                                   data_info.loc[tname].fasta, processed_path))
    print('finish!')
    
    random.shuffle(train_datasets)
    
    print('loading eval_tasks ...')
    #executor = ThreadPoolExecutor(max_workers=5)
    #all_task=[]
    for tname in eval_tasks:
    #    task = executor.submit(load_dataset,tname,all_pdb,processed_path)
    #    all_task.append(task)
    #for future in as_completed(all_task):
    #    (tname, dataset) = future.result()
        #torch.save(dataset, '//data//xuyuyang//BaiduIntern//processed//processed_dataset//'+tname+'_processeddataset.pt')
        eval_datasets.append(load_dataset(tname, data_info.loc[tname].len_active, \
                                           data_info.loc[tname].len_decoy, \
                                               data_info.loc[tname].new_protein_filename, \
                                                   data_info.loc[tname].fasta, processed_path))
    
    print('loading test_tasks ...')
    #executor = ThreadPoolExecutor(max_workers=5)
    #all_task=[]
    for tname in test_tasks:
    #    task = executor.submit(load_dataset,tname,all_pdb,processed_path)
    #    all_task.append(task)
    #for future in as_completed(all_task):
    #    (tname, dataset) = future.result()
        #torch.save(dataset, '//data//xuyuyang//BaiduIntern//processed//processed_dataset//'+tname+'_processeddataset.pt')
        test_datasets.append(load_dataset(tname, data_info.loc[tname].len_active, \
                                           data_info.loc[tname].len_decoy, \
                                               data_info.loc[tname].new_protein_filename, \
                                                   data_info.loc[tname].fasta, processed_path))
    print('finish!')
    end = time.time()
    print('Takes '+str(int((end-start)/3600))+'h '+str(int((end-start)%3600/60))+'m '+str(int((end-start)%3600%60))+'s.')
        #torch.save((train_datasets, test_datasets, num_train_tasks, num_test_tasks), '//data//xuyuyang//BaiduIntern//processed//processed_data.pt')
    return train_datasets, eval_datasets, test_datasets, num_test_tasks

def load_dti_data_lp(num_train_tasks, used_train_task, processed_path):
    start = time.time()
    data_info = pd.read_csv(processed_path+'data_info.csv',index_col='Unnamed: 0')
    train_tasks=list(data_info.loc[data_info.train==1].pname)
    eval_tasks=list(data_info.loc[data_info.train==0].pname)
    test_tasks=list(data_info.loc[data_info.train==-1].pname)
    
    num_test_tasks = len(test_tasks)
    assert num_train_tasks <= len(train_tasks)
    assert used_train_task <= num_train_tasks
    
    train_tasks = train_tasks[:used_train_task]
    random.shuffle(train_tasks)
    
    train_datasets=[]
    test_datasets=[]
    eval_datasets = []
    
    for tname in train_tasks:
        train_datasets.append(load_dataset_lp(data_info.loc[data_info.pname == tname].index.values[0], tname, data_info.loc[data_info.pname == tname].len_active.values[0], \
                                           data_info.loc[data_info.pname == tname].len_decoy.values[0], \
                                               data_info.loc[data_info.pname == tname].new_protein_filename.values[0], \
                                                   data_info.loc[data_info.pname == tname].fasta.values[0], processed_path))
    print('finish!')
    
    random.shuffle(train_datasets)
    
    print('loading eval_tasks ...')
    
    for tname in eval_tasks:
        eval_datasets.append(load_dataset_lp(data_info.loc[data_info.pname == tname].index.values[0], tname, data_info.loc[data_info.pname == tname].len_active.values[0], \
                                           data_info.loc[data_info.pname == tname].len_decoy.values[0], \
                                               data_info.loc[data_info.pname == tname].new_protein_filename.values[0], \
                                                   data_info.loc[data_info.pname == tname].fasta.values[0], processed_path))
    
    print('loading test_tasks ...')
    
    for tname in test_tasks:
        test_datasets.append(load_dataset_lp(data_info.loc[data_info.pname == tname].index.values[0], tname, data_info.loc[data_info.pname == tname].len_active.values[0], \
                                           data_info.loc[data_info.pname == tname].len_decoy.values[0], \
                                               data_info.loc[data_info.pname == tname].new_protein_filename.values[0], \
                                                   data_info.loc[data_info.pname == tname].fasta.values[0], processed_path))
    print('finish!')
    end = time.time()
    print('Takes '+str(int((end-start)/3600))+'h '+str(int((end-start)%3600/60))+'m '+str(int((end-start)%3600%60))+'s.')
        #torch.save((train_datasets, test_datasets, num_train_tasks, num_test_tasks), '//data//xuyuyang//BaiduIntern//processed//processed_data.pt')
    return train_datasets, eval_datasets, test_datasets, num_test_tasks

