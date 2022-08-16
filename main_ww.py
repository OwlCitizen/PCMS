#main.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
import random
import torch
#torch.autograd.set_detect_anomaly(True)
import numpy as np
import pandas as pd
runseed = 42
torch.manual_seed(runseed)
np.random.seed(runseed)
random.seed(runseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(runseed)
import math
import time as timem
from dti_loader import load_dti_data
from meta_model_ww import Meta_model
from tensorboardX import SummaryWriter
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector


data_path='//data//xuyuyang//BaiduIntern//all//'
path = '/data2/xuyuyang/NMI_MetaDTI/model/'
processed_path = '//data//xuyuyang//BaiduIntern//processed_schnet//'

folds_num = 1
PATIENCE = 10
epochs = 600

emb_dim = 128
n_way = 2
m_support = 5
k_query= 128
update_lr = 1e-5
meta_lr = 1e-5
test_update_lr = 1e-5
decay = 1e-15
inner_loop = 5
mlayer = 2
slayer = 6
regular = 0
regular_file = ''
ex_weight = 1e-3
batch_size = 5
mode = 0
psimple = True
add_similarity = True
add_metasimilarity = False
add_selfsupervise = True
batchnorm = True
attention_detach = True
input_model_file = ''
slave_fix = False
if not os.path.exists(input_model_file):
    slave_fix = False
max_pvae_step = 1000
all_pdb = True
tune_disc = False

num_train_tasks = 50
used_train_task = 50
task_batch_size = 5
update_step = 5
update_step_test = 5
epoch_test = 5

#checkpoint = '//home//xuyuyang//BaiduIntern//Models//model//net_params//'+\
#    'test_main_ww_NewRobust_5_128_tbs5_emb128_exweight0.001_lr1e-05_mlr1e-05_tlr1e-05_psimple_addsimilarity_batchnorm_attentiondetach'+\
#        '//fold0//checkpoint.pt'
checkpoint = ''
epoch_check = 1

curtime = timem.strftime('%Y/%m/%d|%H:%M:%S: ',timem.localtime(timem.time()))


DATE = 'test_main_ww_220726_'+str(m_support)+'_'+str(k_query)+'_tbs'+str(task_batch_size)+'_emb'+str(emb_dim)+'_exweight'+str(ex_weight)+'_lr'+str(update_lr)+'_mlr'+str(meta_lr)+'_tlr'+str(test_update_lr)+\
    ('_psimple' if psimple else '')+('_addsimilarity' if add_similarity else '') + ('_addmetasimilarity' if add_metasimilarity else '')+\
        ('_batchnorm' if batchnorm else '') + ('_attentiondetach' if attention_detach else '') + ('_fix' if slave_fix else '')
'''
DATE = 'test'
'''

def save_checkpoint(fold, epoch, patience, patience_, best_params, best_params_, \
                    best_epoch, best_epoch_, best_test_acc, best_test_auc, net, save_path):
    cp_dict = {}
    cp_dict['fold'] = fold
    cp_dict['epoch'] = epoch
    cp_dict['patience'] = patience
    cp_dict['patience_'] = patience_
    cp_dict['params'] = net.state_dict()
    cp_dict['best_params'] = best_params
    cp_dict['best_params_'] = best_params_
    cp_dict['best_epoch'] = best_epoch
    cp_dict['best_epoch_'] = best_epoch_
    cp_dict['best_test_acc'] = best_test_acc
    cp_dict['best_test_auc'] = best_test_auc
    cp_dict['param_dict'] = net.mlearner.param_dict
    torch.save(cp_dict, save_path)    

def load_checkpoint(cp_filepath):
    cp_dict = torch.load(cp_filepath)
    return cp_dict

def mean(array):
    assert len(array)>0
    return sum(array)/len(array)

def printf(text):
    text = str(text)
    file = open(".//results//outs.txt", "a+")
    file.write(curtime+text+'\n')
    file.close()
    file = open(".//results//"+DATE+"_outs.txt", "a+")
    file.write(curtime+text+'\n')
    file.close()
    print(text)
    
def printf_(text = ''):
    text = str(text)
    file = open(".//results//log.txt", "a+")
    file.write(curtime+text+'\n')
    file.close()
    file = open(".//results//outs.txt", "a+")
    file.write(curtime+text+'\n')
    file.close()
    print(text) 

def main():
    printf('\n\n\nRunning '+DATE)
    
    net = Meta_model(emb_dim, n_way, m_support, k_query, update_lr, meta_lr, test_update_lr, update_step,\
                 update_step_test, decay, inner_loop, max_pvae_step, mlayer, slayer, regular,\
                     ex_weight, batch_size, mode, psimple, add_similarity, add_metasimilarity, \
                         add_selfsupervise, regular_file, batchnorm, attention_detach, tune_disc, slave_fix).cuda()
    net.init_params()
        
    if os.path.exists(checkpoint):
        cp_dict = torch.load(checkpoint)
        net.load_state_dict(cp_dict['params'])
        net.mlearner.param_disct = cp_dict['param_dict']
        print('check point at epoch '+str(epoch_check)+' loaded!')
        epoch_start = cp_dict['epoch']
        fold_start = cp_dict['fold']
        cp_flag = True
    else:
        epoch_start = 0
        fold_start = 0
        cp_flag = False
    train_set, eval_set, test_set, num_test_tasks = load_dti_data(num_train_tasks, used_train_task, processed_path)
    
    test_tnames_acc=[]
    test_tnames_auc=[]
    test_tnames_f1=[]
    test_tnames_prec=[]
    test_tnames_recall=[]
    test_tnames_acc0=[]
    test_tnames_auc0=[]
    test_tnames_f10=[]
    test_tnames_prec0=[]
    test_tnames_recall0=[]
    
    test_tnames_acc_=[]
    test_tnames_auc_=[]
    test_tnames_f1_=[]
    test_tnames_prec_=[]
    test_tnames_recall_=[]
    test_tnames_acc0_=[]
    test_tnames_auc0_=[]
    test_tnames_f10_=[]
    test_tnames_prec_=[]
    test_tnames_recall_=[]
    for t in range(len(test_set)):
        test_tnames_acc0.append(test_set[t][0]+'_startacc')
        test_tnames_auc0.append(test_set[t][0]+'_startauc')
        test_tnames_acc.append(test_set[t][0]+'_acc')
        test_tnames_auc.append(test_set[t][0]+'_auc')
        test_tnames_f10.append(test_set[t][0]+'_startf1')
        test_tnames_f1.append(test_set[t][0]+'_f1')
        test_tnames_acc0_.append(test_set[t][0]+'_startacc(anchor)')
        test_tnames_auc0_.append(test_set[t][0]+'_startauc(anchor)')
        test_tnames_acc_.append(test_set[t][0]+'_acc(anchor)')
        test_tnames_auc_.append(test_set[t][0]+'_auc(anchor)')
        test_tnames_f10_.append(test_set[t][0]+'_startf1(anchor)')
        test_tnames_f1_.append(test_set[t][0]+'_f1(anchor)')
        
    eval_tnames_acc = []
    eval_tnames_auc = []
    eval_tnames_acc0 = []
    eval_tnames_auc0 = []
    eval_tnames_f1 = []
    eval_tnames_f10 = []
    
    for t in range(len(eval_set)):
        eval_tnames_acc0.append(eval_set[t][0]+'_startacc_eval')
        eval_tnames_auc0.append(eval_set[t][0]+'_startauc_eval')
        eval_tnames_acc.append(eval_set[t][0]+'_acc_eval')
        eval_tnames_auc.append(eval_set[t][0]+'_auc_eval')
        eval_tnames_f10.append(eval_set[t][0]+'_startf1_eval')
        eval_tnames_f1.append(eval_set[t][0]+'_f1_eval')
    
    test_tnames = test_tnames_acc0 + ['mean_startacc'] + test_tnames_auc0 + \
        ['mean_startauc'] + test_tnames_f10 + ['mean_startf1'] + test_tnames_acc + \
            ['mean_acc'] + test_tnames_auc + ['mean_auc'] + test_tnames_f1 + ['mea_f1'] + \
                test_tnames_acc0_ + ['mean_startacc(anchor)'] + test_tnames_auc0_ + \
                    ['mean_startauc(anchor)'] + test_tnames_f10_ + ['mean_startf1(anchor)'] + \
                        test_tnames_acc_ + ['mean_acc(anchor)'] + test_tnames_auc_ + \
                            ['mean_auc(anchor)'] + test_tnames_f1_ + ['mean_f1(anchor)'] + \
                                eval_tnames_acc0 + ['mean_startacc_eval'] + \
                                    eval_tnames_auc0 + ['mean_startauc_eval'] + \
                                        eval_tnames_f10 + ['mean_startf1_eval'] + \
                                            eval_tnames_acc + ['mean_acc_eval'] + \
                                                eval_tnames_auc + ['mean_auc_eval'] + \
                                                    eval_tnames_f1 + ['mean_f1_eval']
    
    resultframe = pd.DataFrame(np.zeros([1,len(test_tnames)]), columns = test_tnames, index = [10])
    
    fstart=timem.time()
    
    try:
        os.mkdir(path+'net_params//'+DATE+'//')
    except:
        printf('Folder already exists!')
    
    start=timem.time()
    for f in range(fold_start, folds_num):
        try:
            os.mkdir(path+'net_params//'+DATE+'//fold'+str(f)+'//')
        except:
            printf('Folder already exists!')
        
        printf('Now running fold '+str(f)+' ...')
        writer = SummaryWriter(path+'tensorboard//'+DATE+'//fold'+str(f))
        net.set_writer(writer)
        
        printf('Start trainning ...')
        fstart=timem.time()
        patience= cp_dict['patience'] if cp_flag else PATIENCE
        patience_ = cp_dict['patience_'] if cp_flag else PATIENCE
        best_params= cp_dict['best_params'] if cp_flag else []
        best_params_ = cp_dict['best_params_'] if cp_flag else []
        best_epoch = cp_dict['best_epoch'] if cp_flag else 0
        best_epoch_ = cp_dict['best_epoch_'] if cp_flag else 0
        best_test_acc = cp_dict['best_test_acc'] if cp_flag else 0
        best_test_auc = cp_dict['best_test_auc'] if cp_flag else 0
        net.mlearner.param_dict = cp_dict['param_dict'] if cp_flag else {}
        
        for epoch in range(epoch_start, epochs+1):
            net.train()
            total = math.ceil(len(train_set)/task_batch_size)
            left = range(0, len(train_set))
            for step in range(total):
                if len(left) >= task_batch_size:
                    idx = random.sample(left, task_batch_size)
                else:
                    idx = left
                left_ = [i for i in left if i not in idx]
                left = left_
                
                task_batch = [train_set[i] for i in idx]
                sstart=timem.time()
                printf('\nNow running fold '+str(f)+' epoch '+str(epoch)+'\tstep '+str(step)+'/'+str(total)+'\t'+str(int(step*100/total))+'%')
                loss_q = net(task_batch, epoch, total, step)
                
                send=timem.time()
                    
                printf('Saving ...')
                torch.save(net.state_dict(), path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params.pkl')
                torch.save(best_params, path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params(best_acc).pkl')
                torch.save(best_params_, path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params(best_auc).pkl')
                torch.save(net.mlearner.param_dict, path+'net_params//'+DATE+'//fold'+str(f)+'//param_dict.pkl')
                save_checkpoint(f, epoch, patience, patience_, best_params, best_params_, best_epoch, best_epoch_, best_test_acc, best_test_auc, net, path+'net_params//'+DATE+'//fold'+str(f)+'//checkpoint.pt')
                printf('Save success!')
                
                printf('Slave Loss: '+str(loss_q)+'|Patience: '+str(patience)+'|STEP TAKES:'+str(int(((send-sstart)%3600)%60))+'s'\
                              +'|FOLD HAS RUN:'+str(int((send-fstart)/3600))+'h '+str(int(((send-fstart)%3600)/60))+'min '+str(int(((send-fstart)%3600)%60))+'s'\
                                  +'|HAS RUN:'+str(int((send-start)/3600))+'h '+str(int(((send-start)%3600)/60))+'min '+str(int(((send-start)%3600)%60))+'s')
                writer.add_scalar('00.Training Loss', loss_q, (epoch*total+step))
                params=list(net.named_parameters())
                for p in range(len(params)):
                    num=p+2
                    if num<10:
                        num='0'+str(num)
                    else:
                        num=str(num)
                    writer.add_scalar(num+'. '+str(params[p][0]), torch.mean(params[p][1]),(epoch*total+step))
                    if params[p][1].grad is None:
                        printf_('Warning:fold '+str(f)+' epoch '+str(epoch)+' step '+str(step)+' params '+str(p)+'\t'+str(params[p][0])+'\t grad is None!')
                    else:
                        writer.add_scalar(num+'. '+str(params[p][0])+' grad', torch.mean(params[p][1].grad),(epoch*total+step))
            
            train_params = parameters_to_vector(net.parameters()).clone().detach_()
            if epoch%epoch_test == 0:
                print('evaluating ....')
                net.eval()
                acc0_list = []
                roc0_list = []
                f10_list = []
                acc_list = []
                roc_list = []
                f1_list = []
                
                acc0_list_ = []
                roc0_list_ = []
                f10_list_ = []
                acc_list_ = []
                roc_list_ = []
                f1_list_ = []
                total=math.ceil(len(test_set)/task_batch_size)
                for step in range(total):
                    task_batch = test_set[step*task_batch_size:(step+1)*task_batch_size]
                    
                    sstart=timem.time()
                    #printf('\nNow running fold '+str(f)+' epoch '+str(epoch)+'\tstep '+str(step)+'/'+str(total)+'\t'+str(int(step*100/total))+'%')
                    acc0s, roc0s, accs, rocs, f10s, f1s=net.test(task_batch, epoch)
                    #optimizer.zero_grad()           # clear gradients for this training step
                    #loss.backward()                 # backpropagation, compute gradients
                    #optimizer.step()                # apply gradients
                    acc_list+=accs
                    roc_list+=rocs 
                    f1_list+=f1s
                    acc0_list+=acc0s
                    roc0_list+=roc0s
                    f10_list +=f10s
                    
                    acc0s, roc0s, accs, rocs, f10s, f1s=net.test_(task_batch, epoch)
                    #optimizer.zero_grad()           # clear gradients for this training step
                    #loss.backward()                 # backpropagation, compute gradients
                    #optimizer.step()                # apply gradients
                    acc_list_+=accs
                    roc_list_+=rocs 
                    f1_list_+= f1s
                    acc0_list_+=acc0s
                    roc0_list_+=roc0s
                    f10_list_+=f10s
                    
                result = acc0_list+[mean(acc0_list)]+roc0_list+[mean(roc0_list)]+\
                    f10_list+[mean(f10_list)]+acc_list+[mean(acc_list)]+\
                        roc_list+[mean(roc_list)]+f1_list+[mean(f1_list)]+\
                            acc0_list_+[mean(acc0_list_)]+roc0_list_+\
                                [mean(roc0_list_)]+f10_list_+[mean(f10_list_)]+\
                                    acc_list_+[mean(acc_list_)]+roc_list_+\
                                        [mean(roc_list_)]+f1_list_+[mean(f1_list_)]
                
                assert ((parameters_to_vector(net.parameters()).clone().detach_()-train_params)==0).all()
                
                acc0_list = []
                roc0_list = []
                f10_list = []
                acc_list = []
                roc_list = []
                f1_list = []
                
                total=math.ceil(len(eval_set)/task_batch_size)
                for step in range(total):
                    task_batch = eval_set[step*task_batch_size:(step+1)*task_batch_size]
                    
                    sstart=timem.time()
                    #printf('\nNow running fold '+str(f)+' epoch '+str(epoch)+'\tstep '+str(step)+'/'+str(total)+'\t'+str(int(step*100/total))+'%')
                    acc0s, roc0s, accs, rocs, f10s, f1s=net.test(task_batch, epoch, ' (anchor)')
                    #optimizer.zero_grad()           # clear gradients for this training step
                    #loss.backward()                 # backpropagation, compute gradients
                    #optimizer.step()                # apply gradients
                    acc_list+=accs
                    roc_list+=rocs 
                    acc0_list+=acc0s
                    roc0_list+=roc0s
                    f1_list+=f1s
                    f10_list+=f10s
                    
                result+=acc0_list+[mean(acc0_list)]+roc0_list+[mean(roc0_list)]+\
                    f10_list+[mean(f10_list)]+acc_list+[mean(acc_list)]+\
                        roc_list+[mean(roc_list)]+f1_list+[mean(f1_list)]
                
                assert ((parameters_to_vector(net.parameters()).clone().detach_()-train_params)==0).all()
                
                if epoch==10:
                    resultframe.iloc[0]=result
                else:
                    resultframe.loc[epoch]=result
                resultframe.to_csv('.//results//'+DATE+'_results.csv')
                
                if(mean(acc_list)<=best_test_acc):
                    patience -= 1
                    if type(best_params) != list:
                        torch.save(best_params, path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params(best_acc).pkl')
                else:
                    best_test_acc = mean(acc_list)
                    best_params = {'best_params':net.state_dict(), 'param_dict':net.mlearner.param_dict}
                    best_epoch = epoch
                    patience = PATIENCE
                if(mean(roc_list)<=best_test_auc):
                    patience_ -= 1
                    if type(best_params_) != list:
                        torch.save(best_params_, path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params(best_auc).pkl')
                else:
                    best_test_auc = mean(roc_list)
                    best_params_ = {'best_params':net.state_dict(), 'param_dict':net.mlearner.param_dict}
                    best_epoch_ = epoch
                    patience_ = PATIENCE
                
            if patience<=0 and patience<=0:
                printf('Epoch '+str(epoch)+' OVERFIT!')
                break
        
        printf('Trainning finish !')
        
        torch.save(net.state_dict(), path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params.pkl')
        torch.save(best_params, path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params(best_acc).pkl')
        torch.save(best_params_, path+'net_params//'+DATE+'//fold'+str(f)+'//net_'+DATE+'_params(best_auc).pkl')
        torch.save(net.mlearner.param_dict, path+'net_params//'+DATE+'//fold'+str(f)+'//param_dict.pkl')
        writer.close()
        
        
        printf('Start testing ......')
        params = {'best_params':net.state_dict(), 'param_dict':net.mlearner.param_dict}
        
        #with torch.no_grad():
        for tf in range(5): 
            
            torch.manual_seed(tf)
            np.random.seed(tf)
            random.seed(tf)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(tf)
            
            net.eval()
            acc_list=[]
            roc_list=[]
            f1_list=[]
            acc0_list = []
            roc0_list = []
            f10_list = []
            bestacc_list=[]
            bestroc_list=[]
            bestf1_list=[]
            bestacc0_list=[]
            bestroc0_list=[]
            bestf10_list=[]
            best_acc_list=[]
            best_roc_list=[]
            best_f1_list=[]
            best_acc0_list=[]
            best_roc0_list=[]
            best_f10_list=[] 
            
            net.load_state_dict(params)
            total=math.ceil(len(test_set)/task_batch_size)
            for step in range(total):
                task_batch = test_set[step*task_batch_size:(step+1)*task_batch_size]
                sstart=timem.time()
                acc0s, roc0s, accs, rocs, f10s, f1s=net.test(task_batch, 'test0')
                acc_list+=accs
                roc_list+=rocs
                f1_list+=f1s
                acc0_list+=acc0s
                roc0_list+=roc0s
                f10_list+=f10s
                send=timem.time()
            
            printf('=========================RESULT=========================')
            printf('ACCs: '+str(acc_list)+' '+str(mean(acc_list)))
            printf('AUCs: '+str(roc_list)+' '+str(mean(roc_list)))
            printf('F1s: '+str(f1_list)+' '+str(mean(f1_list)))
            printf('ACC0s: '+str(acc0_list)+' '+str(mean(acc0_list)))
            printf('AUC0s: '+str(roc0_list)+' '+str(mean(roc0_list)))
            printf('F10s: '+str(f10_list)+' '+str(mean(f10_list)))
            printf('========================================================')
            content=acc0_list+[mean(acc0_list)]+roc0_list+\
                [mean(roc0_list)]+f10_list+[mean(f10_list)]+acc_list+\
                    [mean(acc_list)]+roc_list+[mean(roc_list)]+f1_list+[mean(f1_list)]
            padding = ['-']*(len(test_tnames) - len(content))
            resultframe.loc['final('+str(tf)+')'] = content + padding
            resultframe.to_csv('.//results//'+DATE+'_finalresults.csv')
            
            if best_epoch!=0:
                net.load_state_dict(best_params)
                net.eval()
                for step in range(total):
                    task_batch = test_set[step*task_batch_size:(step+1)*task_batch_size]
                    sstart=timem.time()
                    #print('\nNow running fold '+str(f)+' epoch '+str(epoch)+'\tstep '+str(step)+'/'+str(total)+'\t'+str(int(step*100/total))+'%')
                    acc0s, roc0s, accs, rocs, f10s, f1s=net.test(task_batch,'test1')
                    #optimizer.zero_grad()           # clear gradients for this training step
                    #loss.backward()                 # backpropagation, compute gradients
                    #optimizer.step()                # apply gradients
                    bestacc_list+=accs
                    bestroc_list+=rocs
                    bestf1_list+=f1s
                    bestacc0_list+=acc0s
                    bestroc0_list+=roc0s
                    bestf10_list+=f10s
                    send=timem.time()
            
                printf('=========================Best Epoch '+str(best_epoch)+'=========================')
                printf('ACCs: '+str(bestacc_list)+' '+str(mean(bestacc_list)))
                printf('AUCs: '+str(bestroc_list)+' '+str(mean(bestroc_list)))
                printf('F1s: '+str(bestf1_list)+' '+str(mean(bestf1_list)))
                printf('ACC0s: '+str(bestacc0_list)+' '+str(mean(bestacc0_list)))
                printf('AUC0s: '+str(bestroc0_list)+' '+str(mean(bestroc0_list)))
                printf('F10s: '+str(bestf10_list)+' '+str(mean(bestf10_list)))
                printf('========================================================')
                resultframe.loc['Best Acc('+str(tf)+')']=bestacc0_list+[mean(bestacc0_list)]+\
                    bestroc0_list+[mean(bestroc0_list)]+bestf10_list+[mean(bestf10_list)]+\
                        bestacc_list+[mean(bestacc_list)]+bestroc_list+[mean(bestroc_list)]+\
                            bestf1_list+[mean(bestf1_list)]+padding
                resultframe.to_csv('.//results//'+DATE+'_finalresults.csv')
            if best_epoch_!=0:
                net.load_state_dict(best_params_)
                net.eval()
                for step in range(total):
                    task_batch = test_set[step*task_batch_size:(step+1)*task_batch_size]
                    sstart=timem.time()
                    #print('\nNow running fold '+str(f)+' epoch '+str(epoch)+'\tstep '+str(step)+'/'+str(total)+'\t'+str(int(step*100/total))+'%')
                    acc0s, roc0s, accs, rocs, f10s, f1s=net.test(task_batch,'test2')
                    #optimizer.zero_grad()           # clear gradients for this training step
                    #loss.backward()                 # backpropagation, compute gradients
                    #optimizer.step()                # apply gradients
                    best_acc_list+=accs
                    best_roc_list+=rocs
                    best_f1_list+=f1s
                    best_acc0_list+=acc0s
                    best_roc0_list+=roc0s
                    best_f10_list+=f10s
                    send=timem.time()
            
                printf('=========================Best Epoch '+str(best_epoch)+'=========================')
                printf('ACCs: '+str(best_acc_list)+' '+str(mean(bestacc_list)))
                printf('AUCs: '+str(best_roc_list)+' '+str(mean(bestroc_list)))
                printf('F1s: '+str(best_f1_list)+' '+str(mean(bestf1_list)))
                printf('ACC0s: '+str(best_acc0_list)+' '+str(mean(bestacc0_list)))
                printf('AUC0s: '+str(best_roc0_list)+' '+str(mean(bestroc0_list)))
                printf('F10s: '+str(best_f10_list)+' '+str(mean(bestf10_list)))
                printf('========================================================')
                resultframe.loc['Best Auc('+str(tf)+')']=best_acc0_list+[mean(best_acc0_list)]+\
                    best_roc0_list+[mean(best_roc0_list)]+best_f10_list+[mean(best_f10_list)]+\
                        best_acc_list+[mean(best_acc_list)]+best_roc_list+[mean(best_roc_list)]+\
                            best_f1_list+[mean(best_f1_list)]+padding
                resultframe.to_csv('.//results//'+DATE+'_finalresults.csv')
            
            
if __name__ == '__main__':
    main()
