#meta_model.py
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from Bio import pairwise2 as pw2
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch_geometric.loader import DataLoader
from utils import FocalLoss, sample_datasets, sample_test_datasets
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from model import ProteinHeteroGNN, ParamDecoder_, ParamVAE_, GIB, SubGraph, SchNet, SchNet_,\
    MySchNet, MySchNet_, GraphDecoder, Subgraph_Disc

beta = 1

def copy_disc(source):
    disc = Subgraph_Disc(source.emb_dim)
    disc = disc.cuda() if source.fc1.weight.data.is_cuda else disc
    vector_to_parameters(parameters_to_vector(source.parameters()).clone().detach_(), disc.parameters())
    return disc

class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(inplace = True),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x

def MI_Est(discriminator, embeddings, positive):
    assert embeddings.shape[0] == positive.shape[0]
    batch_size = embeddings.shape[0]
    shuffle_embeddings = embeddings[torch.randperm(batch_size)]
    joint = discriminator(embeddings,positive)
    margin = discriminator(shuffle_embeddings,positive)
    mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))
    return mi_est

class Meta_learner(nn.Module):
    def __init__(self, sl_pshapes, emb_dim=128, num_layer=2, drop_ratio = 0.5, \
                 simple = False, con_weight = 5, regular = 0, ex_weight = 0.01,\
                     inner_loop = 5, meta_lr = 1e-3, decay = 5e-5, max_pvae_step=1000, input_model_file_ = '', batchnorm=False):
        super(Meta_learner, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.con_weight = con_weight
        self.sl_pshapes = sl_pshapes
        self.simple = simple
        self.max_pvae_step = max_pvae_step
        self.regular = regular
        self.ex_weight = ex_weight
        self.decoder = GraphDecoder()
        self.inner_loop = inner_loop
        self.meta_lr = meta_lr
        self.decay = decay
        
        self.plearner = ProteinHeteroGNN(hidden_channels1=emb_dim, hidden_channels2 = 1024, num_layers=num_layer)#, batchnorm = batchnorm)
        self.gib = GIB(self.emb_dim, self.drop_ratio, self.con_weight, connect = True)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.abn = nn.BatchNorm1d(self.emb_dim)
            self.sabn = nn.BatchNorm1d(self.emb_dim)
            self.rbn = nn.BatchNorm1d(self.emb_dim)
        #self.paramvae1 = ParamVAE_(self.emb_dim, self.sl_pshapes[0])
        #self.paramvae2 = ParamVAE_(self.emb_dim, self.sl_pshapes[1])
        self.param_dict = {}
        self.param_attention_1 = nn.Sequential(
            nn.Linear(2*self.emb_dim, self.emb_dim),
            nn.Softplus(),
            nn.Linear(self.emb_dim, 1),
        )
        self.param_attention_2 = nn.Sequential(
            nn.Linear(2*self.emb_dim, self.emb_dim),
            nn.Softplus(),
            nn.Linear(self.emb_dim, 1),
        )
        self.paramloss_func = nn.SmoothL1Loss(reduction = 'none')
        
        if not self.simple:
            self.masking_criterion = nn.CrossEntropyLoss()
            self.masking_linear = nn.Linear(self.emb_dim, 119)
            
        if self.regular and input_model_file_ != '':
            sample = MySchNet(hidden_channels=emb_dim)
            sample.load_state_dict(torch.load(input_model_file_))
            self.anchor_params = parameters_to_vector(sample.graph_pred_linear.parameters()).cuda().clone().detach_()
            sample = None
            self.regular_lossfunc = nn.MSELoss()
        elif self.regular and input_model_file_ == '':
            raise ValueError('unimplemented regular file name!')
    
    def set_optimizer(self):
        self.meta_optimizer = optim.RMSprop(self.parameters(), lr=self.meta_lr, weight_decay=self.decay)
        #if self.scheduler:
        #    self.meta_scheduler = StepLR(self.meta_optimizer, step_size=5, gamma=0.5)
    
    def msa_loss(self, fastas, residues):
        #emb = torch.cat(residues, dim=0)
        emb = residues
        msa_edge_start = []
        msa_edge_end = []
        starts = []
        start = 0
        for fasta in fastas:
            starts.append(start)
            start+=len(fasta)
        for i in range(len(fastas)-1):
            for j in range(i+1, len(fastas)):
                fasta1 = fastas[i]
                fasta2 = fastas[j]
                align = pw2.align.globalxx(fasta1, fasta2)
                assert len(align[0][0]) == len(align[0][1])
                align1 = align[0][0]
                align2 = align[0][1]
                acount = -1
                bcount = -1
                for r in range(len(align1)):
                    if align1[r]!='-':
                        acount += 1
                    if align2[r]!='-':
                        bcount += 1
                    if align1[r]==align2[r]:
                        msa_edge_start.append(starts[i]+acount)
                        msa_edge_start.append(starts[j]+bcount)
                        msa_edge_end.append(starts[j]+bcount)
                        msa_edge_end.append(starts[i]+acount)
        
        msa_edge_index = torch.cat([torch.tensor(msa_edge_start, dtype=torch.long).unsqueeze(dim = 0), torch.tensor(msa_edge_end, dtype = torch.long).unsqueeze(dim = 0)],dim = 0)
        msa_edge_index = msa_edge_index.cuda() if residues.is_cuda else msa_edge_index
        #print(msa_edge_index)
        msa_loss = self.decoder(emb, msa_edge_index)/len(fastas)
        return msa_loss
    
    def run(self, protein_dataset, slearner_params):
        pbar = tqdm(range(len(protein_dataset)), desc = 'processing pvec')
        pvec = []
        s_pvec = []
        recon_loss = []
        x = []
        residues_emb = []
        fastas = []
        penalties = []
        tnames = []
        for i in pbar:
            tname, pdata, fasta = protein_dataset[i]
            pdata = pdata.cuda()
            fastas.append(fasta)
            node_emb, pnode_edge_index, residue_emb, pvec_ = self.plearner(pdata.x_dict, pdata.edge_index_dict, pdata.edge_attr_dict)
            node_emb, pvec_, s_pvec_, positive_penalty = self.gib(node_emb, pnode_edge_index)
            penalties+=positive_penalty
            residues_emb.append(residue_emb)
            if not self.simple:
                x.append(pdata.x_dict['atom'])
                recon_loss.append(self.decoder(node_emb, pdata.edge_index_dict[('atom','bond','atom')]))
            pvec.append(pvec_.view(1,self.emb_dim))
            s_pvec.append(s_pvec_.view(1, self.emb_dim))
            tnames.append(tname)
            #self.param_dict[tname] = (s_pvec_.squeeze().clone().detach_(), slearner_params[0][i].squeeze().clone().detach_(), slearner_params[1][i].squeeze().clone().detach_())
        #pvec = self.bn(torch.cat(pvec,dim=0))
        pvec = torch.cat(pvec,dim=0)
        s_pvec = torch.cat(s_pvec,dim=0)
        residues_emb = torch.cat(residues_emb)
        if self.batchnorm:
            pvec = self.abn(pvec)
            s_pvec = self.sabn(pvec)
            residues_emb = self.rbn(residues_emb)
        
        for i in range(len(tnames)):
            tname = tnames[i]
            self.param_dict[tname] = (s_pvec[i].squeeze().clone().detach_(), slearner_params[0][i].squeeze().clone().detach_(), slearner_params[1][i].squeeze().clone().detach_())
        
        '''
        param_1 = self.paramvae1.decoder(pvec)
        s_param_1 = self.paramvae1.decoder(s_pvec)
        pvec_1 = self.paramvae1.encoder(param_1)
        s_pvec_1 = self.paramvae1.encoder(s_param_1)
        
        param_2 = self.paramvae2.decoder(pvec)
        s_param_2 = self.paramvae2.decoder(s_pvec)
        pvec_2 = self.paramvae2.encoder(param_2)
        s_pvec_2 = self.paramvae2.encoder(s_param_2)
        '''
        
        def vselect(vector, i):
            return vector[torch.arange(vector.size(0))!=i] 
        
        param_1 = []
        s_param_1 = []
        param_2 = []
        s_param_2 = []
        for i in range(len(protein_dataset)):
            temp = vselect(pvec, i)
            s_temp = vselect(s_pvec, i)
            att_pvec = torch.cat([temp, torch.zeros_like(temp).copy_(pvec[i:i+1])],dim = -1)
            att_s_pvec = torch.cat([s_temp, torch.zeros_like(s_temp).copy_(s_pvec[i:i+1])],dim = -1)
            param_1.append(torch.matmul((self.param_attention_1(att_pvec)).t(),vselect(slearner_params[0], i)))
            param_2.append(torch.matmul((self.param_attention_2(att_pvec)).t(),vselect(slearner_params[1], i)))
            s_param_1.append(torch.matmul((self.param_attention_1(att_s_pvec)).t(),vselect(slearner_params[0], i)))
            s_param_2.append(torch.matmul((self.param_attention_2(att_s_pvec)).t(),vselect(slearner_params[1], i)))
        
        param_1 = torch.cat(param_1, dim = 0)
        s_param_1 = torch.cat(s_param_1, dim = 0)
        param_2 = torch.cat(param_2, dim = 0)
        s_param_2 = torch.cat(s_param_2, dim = 0)
        
        pvae_loss = self.paramloss_func(slearner_params[0], param_1).mean(dim = 1)+self.paramloss_func(slearner_params[1], param_2).mean(dim = 1)
        #pvec_loss = self.paramloss_func(pvec, pvec_1).mean(dim = 1)+self.paramloss_func(pvec, pvec_2).mean(dim = 1)
        pvae_loss_s = self.paramloss_func(slearner_params[0], s_param_1).mean(dim = 1)+self.paramloss_func(slearner_params[1], s_param_2).mean(dim = 1)
        #pvec_loss_s = self.paramloss_func(s_pvec, s_pvec_1).mean(dim = 1)+self.paramloss_func(s_pvec, s_pvec_2).mean(dim = 1)
        
        msa_loss = self.msa_loss(fastas, residues_emb)
        
        loss = (pvae_loss, pvae_loss_s, torch.cat(penalties), msa_loss)
        
        if self.regular:
            assert param_2.shape[1] == self.anchor_params.shape[-1]
            assert s_param_2.shape[1] == self.anchor_params.shape[-1]
            regular_loss = self.paramloss_func(param_2, torch.zeros(param_2.shape).copy_(self.anchor_params).cuda()).mean(dim = 1)
            regular_loss_s = self.paramloss_func(s_param_2, torch.zeros(param_2.shape).copy_(self.anchor_params).cuda()).mean(dim = 1)
            loss += (regular_loss, regular_loss_s)
        else:
            loss += (torch.empty(len(protein_dataset)), torch.empty(len(protein_dataset)))
        if not self.simple:
            #pvae_loss += self.ex_weight * recon_loss
            x = torch.cat(x, dim=0)
            mask_num = random.sample(range(0,node_emb.size()[0]), len(protein_dataset))
            pred_emb = self.masking_linear(node_emb[mask_num])
            self_loss = self.masking_criterion(pred_emb.double(), x[mask_num,0].long())
            pred_emb = None
            loss += (torch.cat(recon_loss), self_loss)
        else:
            loss += (torch.empty(len(protein_dataset)), torch.empty(len(protein_dataset)))
        
        return ((param_1, param_2),(s_param_1, s_param_2)), (pvec, s_pvec), loss
    
    def forward(self, protein_dataset, slearner_params):#, m_disc):
        l_pvae_loss = 100
        pvae_loss_diff = 100
        #for i in range(len(protein_dataset)):
        #    protein_dataset[i] = protein_dataset[i].cuda()
        loss_count = 0
        for count in range(self.max_pvae_step):
            pvae_loss = 0
            _, (pemb, pemb_s), pvae_losses = self.run(protein_dataset, slearner_params)
            #writer.......
            pvae_loss += (pvae_losses[0]+pvae_losses[2]+self.ex_weight*(pvae_losses[1]+pvae_losses[3])+self.ex_weight*(pvae_losses[4]+pvae_losses[5]))
            if self.regular:
                pvae_loss += self.ex_weight * (pvae_losses[6]+pvae_losses[7])
            if not self.simple:
                pvae_loss+= self.ex_weight * (pvae_losses[8]+pvae_losses[9])
            '''
            for _ in range(self.inner_loop):
                optimizer_local = torch.optim.Adam(m_disc.parameters(),lr=self.meta_lr, weight_decay=self.decay)
                optimizer_local.zero_grad()
                local_loss = ((0.1*self.ex_weight) if self.mdisc_flag else 1.0) * - MI_Est(m_disc, embeddings.clone().detach_(), positive.clone().detach_())
                local_loss.backward()#retain_graph = True)
                optimizer_local.step()
            mi_loss = MI_Est(m_disc, pemb, pemb_s)
            self.m_disc.save_params()
            self.mdisc_flag = self.m_disc.tune_disc(mi_loss, self.mdisc_flag)
            '''
            if count == 0:
                l_pvae_loss = pvae_loss.clone().detach_()
            else:
                pvae_loss_diff = torch.abs_(l_pvae_loss - pvae_loss.clone().detach_())
                l_pvae_loss = pvae_loss.clone().detach_()
            
            #pvae_loss += ex_weight * msa_loss
            self.meta_optimizer.zero_grad()
            pvae_loss.backward()
            self.meta_optimizer.step()
            print('\tPvae step:'+str(count)+'|pvae loss: '+str(pvae_loss.clone().detach_())+'|pvae_loss_diff:'+str(pvae_loss_diff))
            pvae_loss = None
            if pvae_loss_diff <= 1e-5:
                loss_count+=1
            else:
                loss_count = 0
            if loss_count >= 3:
                print('\tpvae converge at step '+str(count))
                break
        if self.scheduler:
            self.meta_scheduler.step()
    
    def test(self, pdata):
        node_emb, pnode_edge_index, _, _ = self.plearner(pdata.x_dict, pdata.edge_index_dict, pdata.edge_attr_dict)
        _, _, s_pvec_, _ = self.gib(node_emb, pnode_edge_index)
        if self.batchnorm:
            s_pvec_ = self.sabn(s_pvec_)
        base_pvecs = []
        base_params_1 = []
        base_params_2 = []
        for key in self.param_dict.keys():
            base_pvecs.append(self.param_dict[key][0].unsqueeze(dim = 0))
            base_params_1.append(self.param_dict[key][1].unsqueeze(dim = 0))
            base_params_2.append(self.param_dict[key][2].unsqueeze(dim = 0))
        base_pvecs = torch.cat(base_pvecs, dim = 0)
        base_params_1 = torch.cat(base_params_1, dim = 0)
        base_params_2 = torch.cat(base_params_2, dim = 0)
        
        att_pvec = torch.cat([base_pvecs, torch.zeros_like(base_pvecs).copy_(s_pvec_.view(1,self.emb_dim))] ,dim=1)
        param_1 = torch.matmul((self.param_attention_1(att_pvec)).t(), base_params_1)
        param_2 = torch.matmul((self.param_attention_2(att_pvec)).t(), base_params_2)
        
        return (param_1.squeeze(), param_2.squeeze())
        
class Meta_model(nn.Module):
    def __init__(self, emb_dim, n_way, m_support, k_query, update_lr, meta_lr, test_update_lr, update_step=5,\
                 update_step_test = 10, decay=5e-15, inner_loop = 5, max_pvae_step = 1000, mlayer = 2, \
                     slayer = 6, regular=0, ex_weight = 0.001, batch_size = 5, mode = 0, psimple = True, \
                         add_similarity = True, add_metasimilarity = False, add_selfsupervise = True, regular_file = '', \
                             batchnorm = False, attention_detach = True, tune_disc = True, slave_fix = True):
        super(Meta_model, self).__init__()
        self.emb_dim = emb_dim
        
        self.n_way=n_way
        self.m_support=m_support
        self.k_query=k_query
        
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.test_update_lr = test_update_lr
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.decay = decay
        self.inner_loop = inner_loop
        self.max_pvae_step = max_pvae_step
        self.mlayer = mlayer
        self.slayer = slayer
        self.regular = regular
        self.ex_weight = ex_weight
        self.batch_size = batch_size
        self.psimple = psimple
        self.mode = mode
        self.tune_disc = tune_disc
        assert self.mode in [0,1]
        self.loss_func = nn.CrossEntropyLoss(reduction = 'sum')
        
        if regular>0 and regular_file == '':
            raise ValueError('error regular file!')
        
        self.slearner = SubGraph(num_emb = self.emb_dim, num_layer = self.slayer)
        self.sl_pshapes = (parameters_to_vector(self.slearner.gib.graph_assignment.parameters()).shape[-1], \
                           parameters_to_vector(self.slearner.gib.graph_pred_linear.parameters()).shape[-1])
        self.mlearner = Meta_learner(sl_pshapes = self.sl_pshapes, emb_dim=self.emb_dim, \
                                     num_layer=self.mlayer, simple = self.psimple, regular = self.regular, \
                                         ex_weight = self.ex_weight, inner_loop = self.inner_loop, \
                                             meta_lr = self.meta_lr, decay = self.decay,\
                                                 max_pvae_step=self.max_pvae_step, \
                                                     input_model_file_ = regular_file, \
                                                         batchnorm=batchnorm)
        
        self.slave_fix = slave_fix
        self.add_similarity = add_similarity
        self.add_metasimilarity = add_metasimilarity
        self.add_selfsupervise = add_selfsupervise
        self.attention_detach = attention_detach
        if self.add_similarity:
            self.Attention = attention(2*self.emb_dim)
        if self.add_metasimilarity:
            self.meta_attention = attention(self.emb_dim)
        if self.add_selfsupervise:
            self.masking_criterion = nn.CrossEntropyLoss(reduction = 'sum')
            self.masking_linear = nn.Linear(self.emb_dim, 119)
            
        model_param_group = []
        model_param_group.append({"params": self.slearner.parameters()})
        if self.add_selfsupervise:
            model_param_group.append({"params": self.masking_linear.parameters()})
        if self.add_similarity:
            model_param_group.append({"params": self.Attention.parameters()})
        self.optimizer_s = optim.RMSprop(model_param_group, lr=self.update_lr, weight_decay=self.decay)
        self.optimizer = optim.RMSprop(model_param_group, lr=self.meta_lr, weight_decay=self.decay)
        
        meta_model_param_group = []
        meta_model_param_group.append({"params": self.mlearner.parameters()})
        if self.add_selfsupervise:
            meta_model_param_group.append({"params": self.masking_linear.parameters()})
        if self.add_metasimilarity:
            meta_model_param_group.append({"params": self.meta_attention.parameters()})
        self.optimizer_ = optim.RMSprop(meta_model_param_group, lr=meta_lr, weight_decay=decay)
        self.mlearner.set_optimizer()#self.meta_lr, self.decay)
        
        self.m_disc = Subgraph_Disc(self.emb_dim)
        self.mdisc_flag = False
        self.s_disc = Subgraph_Disc(self.emb_dim)
        self.sdisc_flag = False
        #self.test_sdisc_flag = False
        self.f_disc = Subgraph_Disc(self.emb_dim)
        self.fdisc_flag = False
        
        self.graph_decoder = GraphDecoder()
        
        self.writer = ''
    
    def init_params(self):
        for p in list(self.named_parameters()):
            if 'weight' in p[0] and len(p[1].shape)>1:
                torch.nn.init.kaiming_normal_(p[1])
            if 'bias' in p[0]:
                p[1].data.fill_(0)
    
    def slave_pretrain(self, input_model_file):
        self.slearner.gnn.load_state_dict(torch.load(input_model_file))
    
    def set_writer(self, writer):
        self.writer = writer
    
    def update_params(self, loss, update_lr, retain=False):
        grads = torch.autograd.grad(loss, self.slearner.parameters(), retain_graph = retain)#, allow_unused=True)
        params = parameters_to_vector(self.slearner.parameters()) - parameters_to_vector(grads) * update_lr
        grads = None
        return params
        
    def load_params(self, params):
        vector_to_parameters(params[0].squeeze(), self.slearner.gib.graph_assignment.parameters())
        vector_to_parameters(params[1].squeeze(), self.slearner.gib.graph_pred_linear.parameters())
    
    def train_slave(self, loader, extra = False):
        preds = []
        labels = []
        pre_losses = 0
        recon_losses = 0
        self_losses = 0
        graph_embs = []
        subgraph_embs = []
        penalties = []
        for step, batch in enumerate(loader):
            batch = batch.cuda()
            pred, node_emb, graph_emb, subgraph_emb, positive_penalties = self.slearner(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            penalties += positive_penalties
            graph_embs.append(graph_emb)
            subgraph_embs.append(subgraph_emb)
            preds.append(pred)
            labels.append(torch.cat([batch.y, batch.y],dim = 0))
            pre_loss = self.loss_func(pred, torch.cat([batch.y, batch.y],dim = 0))
            pre_losses += pre_loss
            
            recon_loss = self.graph_decoder(node_emb, batch.edge_index, batch.batch, mean = False)
            recon_losses += recon_loss
            
            if self.add_selfsupervise:
                mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                pred_emb = self.masking_linear(node_emb[mask_num])
                self_loss = self.masking_criterion(pred_emb.double(), batch.x[mask_num,0].long())
                self_losses += self_loss
        preds = torch.cat(preds, dim = 0).clone().detach_()
        labels = torch.cat(labels, dim = 0).clone().detach_()
        acc = sum(preds.argmax(dim = 1) == labels)/len(preds)
        auc = roc_auc_score(labels.squeeze().detach().cpu().numpy(), F.softmax(preds,dim = 1)[:,1].squeeze().detach().cpu().numpy())
        f1 = f1_score(labels.squeeze().detach().cpu().numpy(), preds.argmax(dim = 1).squeeze().detach().cpu().numpy())
        prec = precision_score(labels.squeeze().detach().cpu().numpy(), preds.argmax(dim = 1).squeeze().detach().cpu().numpy())
        recall = recall_score(labels.squeeze().detach().cpu().numpy(), preds.argmax(dim = 1).squeeze().detach().cpu().numpy())
        
        pre_losses = pre_losses/len(preds)
        recon_losses = recon_losses/len(preds)
        self_losses = self_losses/len(preds)
        
        return acc.clone().detach_().cpu().item(), auc, f1, pre_losses, recon_losses, self_losses, sum(penalties)/len(penalties), torch.cat(graph_embs, dim = 0), torch.cat(subgraph_embs, dim = 0)
    
    def train_slave_(self, params, loader):
        preds = []
        labels = []
        pre_losses = 0
        recon_losses = 0
        self_losses = 0
        graph_embs = []
        subgraph_embs = []
        penalties = []
        for step, batch in enumerate(loader):
            batch = batch.cuda()
            pred, node_emb, graph_emb, subgraph_emb, positive_penalty = self.slearner.forward_(params, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            penalties += positive_penalty
            graph_embs.append(graph_emb)
            subgraph_embs.append(subgraph_emb)
            preds.append(pred)
            labels.append(torch.cat([batch.y, batch.y],dim = 0))
            pre_loss = self.loss_func(pred, torch.cat([batch.y, batch.y],dim = 0))
            pre_losses += pre_loss
            
            recon_loss = self.graph_decoder(node_emb, batch.edge_index, batch.batch, mean = False)
            recon_losses += recon_loss
            
            if self.add_selfsupervise:
                mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                pred_emb = self.masking_linear(node_emb[mask_num])
                self_loss = self.masking_criterion(pred_emb.double(), batch.x[mask_num,0].long())
                self_losses += self_loss
        preds = torch.cat(preds, dim = 0).clone().detach_()
        labels = torch.cat(labels, dim = 0).clone().detach_()
        acc = sum(preds.argmax(dim = 1) == labels)/len(preds)
        auc = roc_auc_score(labels.squeeze().detach().cpu().numpy(), F.softmax(preds,dim = 1)[:,1].squeeze().detach().cpu().numpy())
        f1 = f1_score(labels.squeeze().detach().cpu().numpy(), preds.argmax(dim = 1).squeeze().detach().cpu().numpy())
        pre_losses = pre_losses/len(preds)
        recon_losses = recon_losses/len(preds)
        self_losses = self_losses/len(preds)
        
        return acc.cpu().item(), auc, f1, pre_losses, recon_losses, self_losses, sum(penalties)/len(penalties), torch.cat(graph_embs, dim = 0), torch.cat(subgraph_embs, dim = 0)
    
    def run_slave(self, task_batch_size, support_loaders, query_loaders, old_params, epoch, total, step, k):
        
        tasks_emb = []
        query_losses = []
        
        qgraph_embses = []
        qsubgraph_embses = []
        
        qaccs = []
        qaucs = []
        qf1s = []
        qpres = []
        qrecons = []
        qselfs = []
        qpenalties = []
        
        self.s_disc.save_params()
        for task in range(task_batch_size):
            tname, support_loader = support_loaders[task]
            tname, query_loader = query_loaders[task]
            support_loss = 0
            query_loss = 0
            
            #===========================train support set =================================
            sacc, sauc, sf1, spre_loss, srecon_loss, sself_loss, sppenalties, sgraph_embs, ssubgraph_embs = self.train_slave(support_loader)
            tasks_emb.append(torch.cat([sgraph_embs, ssubgraph_embs], dim=1).mean(dim=0).view(1,2*self.emb_dim))
            
            self.s_disc.reset_params()
            for _ in range(self.inner_loop):
                optimizer_local = torch.optim.Adam(self.s_disc.parameters(),lr=self.meta_lr, weight_decay=self.decay)
                optimizer_local.zero_grad()
                local_loss = ((0.1*self.ex_weight) if self.sdisc_flag else 1.0) * - MI_Est(self.s_disc, sgraph_embs.clone().detach_(), ssubgraph_embs.clone().detach_())
                local_loss.backward()#retain_graph = True)
                optimizer_local.step()
            smi_loss = MI_Est(self.s_disc, sgraph_embs, ssubgraph_embs)
            
            sgraph_embs = ssubgraph_embs = None
            
            support_loss += (spre_loss + self.ex_weight * (srecon_loss + sself_loss + sppenalties + beta * smi_loss))
            
            out = 'EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|'+str(task)+'.'+tname+':\t'
            out += 'support loss:'+str(round(spre_loss.item(),2))
            out += '|reconl:'+str(round(srecon_loss.item(),2))
            if self.add_selfsupervise:
                out += '|selfl:'+str(round(sself_loss.item(),2))
            out += '|ppenalty:'+str(round(sppenalties.item(),2))
            out += '|mil:'+str(round(smi_loss.item(),2))
            print(out)
            
            new_params = self.update_params(support_loss, update_lr = self.update_lr, retain=True)
            vector_to_parameters(new_params, self.slearner.parameters())
            
            self.s_disc.reset_params()
            #self.optimizer_s.zero_grad()
            #support_loss.backward(retain_graph = True)
            #self.optimizer_s.step()
            
            sacc = sauc = sf1 = spre_loss = srecon_loss = sself_loss = sppenalties = sgraph_embs = ssubgraph_embs = support_loss = new_params = None
            
            
            #==========================train query set====================================
            qacc, qauc, qf1, qpre_loss, qrecon_loss, qself_loss, qppenalties, qgraph_embs, qsubgraph_embs = self.train_slave(query_loader)
            #print(list(self.slearner.gib.graph_assignment.parameters()))
            qgraph_embses.append(qgraph_embs)
            qsubgraph_embses.append(qsubgraph_embs)
            query_loss += (qpre_loss + self.ex_weight * (qrecon_loss + qself_loss + qppenalties))
            query_losses.append(query_loss)
            
            qaccs.append(qacc)
            qaucs.append(qauc)
            qf1s.append(qf1)
            qpres.append(qpre_loss)
            qrecons.append(qrecon_loss)
            qselfs.append(qself_loss)
            qpenalties.append(qppenalties)
            
            out = 'EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|'+str(task)+'.'+tname+':\t'
            out += 'query loss:'+str(round(qpre_loss.item(),2))
            out += '|reconl:'+str(round(qrecon_loss.item(),2))
            if self.add_selfsupervise:
                out += '|selfl:'+str(round(qself_loss.item(),2))
            out += '|ppenalty:'+str(round(qppenalties.item(),2))
            print(out)
            
            qacc = qauc = qf1 = qpre_loss = qrecon_loss = qself_loss = qppenalties = qgraph_embs = qsubgraph_embs = query_loss = None
            
            vector_to_parameters(old_params.clone(), self.slearner.parameters())
        
        qmi_losses = []
        qlocal_losses = []
        for _ in range(self.inner_loop):
            local_loss = 0
            optimizer_local = torch.optim.Adam(self.s_disc.parameters(),lr=self.meta_lr, weight_decay=self.decay)
            optimizer_local.zero_grad()
            for i in range(len(qgraph_embses)):
                local_loss += ((0.1*self.ex_weight) if self.sdisc_flag else 1.0) * (- MI_Est(self.s_disc, qgraph_embses[i].clone().detach_(), qsubgraph_embses[i].clone().detach_()))
            local_loss = local_loss/len(qgraph_embses)
            local_loss.backward()#retain_graph = True)
            optimizer_local.step()
            qlocal_losses.append(local_loss.clone().detach_().item())
        self.s_disc.save_params()
        for i in range(len(qgraph_embses)):
            qmi_losses.append(MI_Est(self.s_disc, qgraph_embses[i], qsubgraph_embses[i]).unsqueeze(dim = 0))
        qmi_losses = torch.cat(qmi_losses).squeeze()
        self.sdisc_flag = self.s_disc.tune_disc(torch.mean(qmi_losses), self.sdisc_flag) if self.tune_disc else self.sdisc_flag
        print('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|MI_Est loss:'+str(qmi_losses.clone().detach_().cpu()))
        
        qgraph_embses = qsubgraph_embses = None
        
        query_losses = torch.cat(query_losses)
        #print(query_losses)
        #print(torch.cat(qmi_losses).squeeze())
        
        query_losses += self.ex_weight * beta * qmi_losses
        
        self.writer.add_scalar('0.Train query acc',sum(qaccs)/len(qaccs),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query auc',sum(qaucs)/len(qaucs),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query f1',sum(qf1s)/len(qf1s),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query pred loss', sum(qpres)/len(qpres), epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query recon loss',sum(qrecons)/len(qrecons),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query self loss',sum(qselfs)/len(qselfs),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query positive penalty',sum(qpenalties)/len(qpenalties),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query mi loss',sum(qmi_losses)/len(qmi_losses),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('0.Train query slave disc loss',sum(qlocal_losses)/len(qlocal_losses),epoch*total*self.update_step+step*self.update_step+k)
        
        tasks_emb = torch.cat(tasks_emb, dim = 0).view(task_batch_size, 2*self.emb_dim)
        if self.attention_detach:
            tasks_emb = tasks_emb.detach()
        if self.add_similarity:
            loss_q = torch.sum(self.Attention(tasks_emb) * query_losses)
        else:
            loss_q = torch.sum(query_losses)/task_batch_size
        #print('loss_q shape:'+str(loss_q.shape))
        
        print('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|Query Loss:'+str(loss_q))
        self.writer.add_scalar('0.Train query loss', loss_q, epoch*total*self.update_step+step*self.update_step+k)
        
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()
        
        return loss_q.clone().detach_()
    
    def run_slave_finetune__(self, task_batch_size, query_loaders, epoch, total, step, k):
        slearner_params1 = []
        slearner_params2 = []
        pbar =tqdm(range(task_batch_size))
        for t in pbar:
            
            ft_disc = copy_disc(self.s_disc).cuda()
            ft_disc.reset_tunecounter()
            ftdisc_flag = False
            
            tname, query_loader = query_loaders[t]
            pbar.set_description('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|finetuning on '+tname)
            
            _, _, _, fpre_loss, frecon_loss, fself_loss, fppenalties, fgraph_embs, fsubgraph_embs = self.train_slave(query_loader)
            '''
            for _ in range(self.inner_loop):
                optimizer_local = optim.RMSprop(ft_disc.parameters(), lr = self.meta_lr)
                optimizer_local.zero_grad()
                local_loss = ((0.1*self.ex_weight) if ftdisc_flag else 1.0) * - MI_Est(ft_disc, fgraph_embs.clone().detach_(), fsubgraph_embs.clone().detach_())
                local_loss.backward()#retain_graph = True)
                optimizer_local.step()
            fmi_loss = MI_Est(ft_disc, fgraph_embs, fsubgraph_embs)
            '''
            finetune_loss = (fpre_loss + self.ex_weight * (frecon_loss + fself_loss + fppenalties))
            #ftdisc_flag = ft_disc.tune_disc(fmi_loss, ftdisc_flag)
            #query_losses.append(query_loss.view(1))
            
            grads1 = parameters_to_vector(torch.autograd.grad(finetune_loss, self.slearner.gib.graph_assignment.parameters(), retain_graph=True)).detach_()
            grads2 = parameters_to_vector(torch.autograd.grad(finetune_loss, self.slearner.gib.graph_pred_linear.parameters(), retain_graph=False)).detach_()
            #sample = SchNet(hidden_channels=self.emb_dim)
            slearner_params1.append((parameters_to_vector(self.slearner.gib.graph_assignment.parameters()).clone().detach_() - grads1 * self.meta_lr).clone().detach_().unsqueeze(dim=0))
            slearner_params2.append((parameters_to_vector(self.slearner.gib.graph_pred_linear.parameters()).clone().detach_() - grads2 * self.meta_lr).clone().detach_().unsqueeze(dim=0))
            
            fpre_loss = frecon_loss = fself_loss = fppenalties = fgraph_embs = fsubgraph_embs = finetune_loss = grads1 = grads2 = None
        
        return torch.cat(slearner_params1, dim = 0), torch.cat(slearner_params2, dim = 0)
    
    def run_slave_finetune_(self, task_batch_size, query_loaders, epoch, total, step, k):
        slearner_params1 = []
        slearner_params2 = []
        pbar =tqdm(range(task_batch_size))
        for t in pbar:
            
            ft_disc = copy_disc(self.s_disc).cuda()
            ft_disc.reset_tunecounter()
            ftdisc_flag = False
            
            tname, query_loader = query_loaders[t]
            pbar.set_description('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|finetuning on '+tname)
            
            _, _, _, fpre_loss, frecon_loss, fself_loss, fppenalties, fgraph_embs, fsubgraph_embs = self.train_slave(query_loader)
            for _ in range(self.inner_loop):
                optimizer_local = optim.RMSprop(ft_disc.parameters(), lr = self.meta_lr)
                optimizer_local.zero_grad()
                local_loss = ((0.1*self.ex_weight) if ftdisc_flag else 1.0) * - MI_Est(ft_disc, fgraph_embs.clone().detach_(), fsubgraph_embs.clone().detach_())
                local_loss.backward()#retain_graph = True)
                optimizer_local.step()
            fmi_loss = MI_Est(ft_disc, fgraph_embs, fsubgraph_embs)
            finetune_loss = (fpre_loss + self.ex_weight * (beta * fmi_loss + frecon_loss + fself_loss + fppenalties))
            ftdisc_flag = ft_disc.tune_disc(fmi_loss, ftdisc_flag) if self.tune_disc else ftdisc_flag
            #query_losses.append(query_loss.view(1))
            
            grads1 = parameters_to_vector(torch.autograd.grad(finetune_loss, self.slearner.gib.graph_assignment.parameters(), retain_graph=True)).detach_()
            grads2 = parameters_to_vector(torch.autograd.grad(finetune_loss, self.slearner.gib.graph_pred_linear.parameters(), retain_graph=False)).detach_()
            #sample = SchNet(hidden_channels=self.emb_dim)
            slearner_params1.append((parameters_to_vector(self.slearner.gib.graph_assignment.parameters()).clone().detach_() - grads1 * self.meta_lr).clone().detach_().unsqueeze(dim=0))
            slearner_params2.append((parameters_to_vector(self.slearner.gib.graph_pred_linear.parameters()).clone().detach_() - grads2 * self.meta_lr).clone().detach_().unsqueeze(dim=0))
            
            fpre_loss = frecon_loss = fself_loss = fppenalties = fgraph_embs = fsubgraph_embs = finetune_loss = grads1 = grads2 = None
        
        return torch.cat(slearner_params1, dim = 0), torch.cat(slearner_params2, dim = 0)
    
    def run_slave_finetune(self, task_batch_size, query_loaders, epoch, total, step, k):
        slearner_params1 = []
        slearner_params2 = []
        pbar =tqdm(range(task_batch_size))
        old_params = parameters_to_vector(self.slearner.parameters()).clone().detach_()
        for t in pbar:
            
            ft_disc = copy_disc(self.s_disc).cuda()
            ft_disc.reset_tunecounter()
            ftdisc_flag = False
            
            tname, query_loader = query_loaders[t]
            for kf in range(self.update_step_test):
                pbar.set_description('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|FinetuneStep:'+str(kf)+'|finetuning on '+tname)
                
                _, _, _, fpre_loss, frecon_loss, fself_loss, fppenalties, fgraph_embs, fsubgraph_embs = self.train_slave(query_loader)
                for _ in range(self.inner_loop):
                    optimizer_local = optim.RMSprop(ft_disc.parameters(), lr = self.meta_lr)
                    optimizer_local.zero_grad()
                    local_loss = ((0.1*self.ex_weight) if ftdisc_flag else 1.0) * - MI_Est(ft_disc, fgraph_embs.clone().detach_(), fsubgraph_embs.clone().detach_())
                    local_loss.backward()#retain_graph = True)
                    optimizer_local.step()
                fmi_loss = MI_Est(ft_disc, fgraph_embs, fsubgraph_embs)
                finetune_loss = (fpre_loss + self.ex_weight * (beta * fmi_loss + frecon_loss + fself_loss + fppenalties))
                ftdisc_flag = ft_disc.tune_disc(fmi_loss, ftdisc_flag) if self.tune_disc else ftdisc_flag
                
                model_param_group = []
                model_param_group.append({"params": self.slearner.gib.graph_assignment.parameters()})
                model_param_group.append({"params": self.slearner.gib.graph_pred_linear.parameters()})
                optimizer = optim.RMSprop(model_param_group, lr = self.meta_lr, weight_decay=self.decay)
                optimizer.zero_grad()
                finetune_loss.backward()
                optimizer.step()
                
                fpre_loss = frecon_loss = fself_loss = fppenalties = fgraph_embs = fsubgraph_embs = finetune_loss = model_param_group = optimizer = None
                
            slearner_params1.append((parameters_to_vector(self.slearner.gib.graph_assignment.parameters()).clone().detach_()).squeeze().unsqueeze(dim = 0))
            slearner_params2.append((parameters_to_vector(self.slearner.gib.graph_pred_linear.parameters()).clone().detach_()).squeeze().unsqueeze(dim = 0))
            vector_to_parameters(old_params.clone(), self.slearner.parameters())
            
        return torch.cat(slearner_params1, dim = 0), torch.cat(slearner_params2, dim = 0)
        
    def train_master_finetune(self, slearner_params1, slearner_params2, protein_dataset, epoch, total, step, k):
        
        paramses, pvecs, plosses = self.mlearner.run(protein_dataset, (slearner_params1, slearner_params2))
        
        pvae_loss, pvae_loss_s, pppenalties, msaloss = plosses[0], plosses[1], plosses[2], plosses[3]
        mloss = (pvae_loss+pvae_loss_s+self.ex_weight*(pppenalties + msaloss))# + pvec_loss + pvec_loss_s))
        
        out = 'EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|'
        out += 'pvae loss:'+str(round((sum(pvae_loss)/len(pvae_loss)).item(),2))
        #out += '|pvec loss:'+str(round((sum(pvec_loss)/len(pvec_loss)).item(),2))
        out += '|sub pvae loss:'+str(round((sum(pvae_loss_s)/len(pvae_loss_s)).item(),2))
        #out += '|sub pvec loss:'+str(round((sum(pvec_loss_s)/len(pvec_loss_s)).item(),2))
        out += '|penalties:'+str(round((sum(pppenalties)/len(pppenalties)).item(),2))
        out += '|msaloss:'+str(round(msaloss.item(),2))
        
        
        self.writer.add_scalar('00.Train pvae loss',sum(pvae_loss)/len(pvae_loss),epoch*total*self.update_step+step*self.update_step+k)
        #self.writer.add_scalar('00.Train pvec loss',sum(pvec_loss)/len(pvec_loss),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train sub pvae loss',sum(pvae_loss_s)/len(pvae_loss_s),epoch*total*self.update_step+step*self.update_step+k)
        #self.writer.add_scalar('00.Train sub pvec loss',sum(pvec_loss_s)/len(pvec_loss_s),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train protein penalties',sum(pppenalties)/len(pppenalties),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train msa loss', msaloss, epoch*total*self.update_step+step*self.update_step+k)
        
        pvae_loss = pvae_loss_s = pppenalties = msaloss = None
        
        plocal_losses = []
        for _ in range(self.inner_loop):
            optimizer_local = torch.optim.Adam(self.m_disc.parameters(),lr=self.meta_lr, weight_decay=self.decay)
            optimizer_local.zero_grad()
            local_loss = ((0.1 * self.ex_weight) if self.mdisc_flag else 1.0) * - MI_Est(self.m_disc, pvecs[0].clone().detach_(), pvecs[1].clone().detach_())
            local_loss.backward()#retain_graph = True)
            optimizer_local.step()
            plocal_losses.append(local_loss.clone().detach_().item())
        pmi_loss = MI_Est(self.m_disc, pvecs[0], pvecs[1])
        mloss += self.ex_weight * beta * pmi_loss
        self.m_disc.save_params()
        self.mdisc_flag = self.m_disc.tune_disc(pmi_loss, self.mdisc_flag) if self.tune_disc else self.mdisc_flag
        
        print('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|Protein MI Est Loss:'+str(pmi_loss))
        self.writer.add_scalar('00.Train protein mi loss', pmi_loss, epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train protein master disc loss', sum(plocal_losses)/len(plocal_losses), epoch*total*self.update_step+step*self.update_step+k)
        
        if self.regular:
            regular_loss, regular_loss_s = plosses[6], plosses[7]
            self.writer.add_scalar('00.Train protein regular loss',sum(regular_loss)/len(regular_loss),epoch*total*self.update_step+step*self.update_step+k)
            self.writer.add_scalar('00.Train sub protein regular loss',sum(regular_loss_s)/len(regular_loss_s),epoch*total*self.update_step+step*self.update_step+k)
            mloss += self.ex_weight * (regular_loss + regular_loss_s)
            regular_loss = regular_loss_s = None
        if not self.psimple:
            precon_loss, pself_loss = plosses[8], plosses[9]
            self.writer.add_scalar('00.Train protein recon loss',sum(precon_loss)/len(precon_loss),epoch*total*self.update_step+step*self.update_step+k)
            self.writer.add_scalar('00.Train protein self loss',sum(pself_loss)/len(pself_loss),epoch*total*self.update_step+step*self.update_step+k)
            mloss += self.ex_weight * (precon_loss + pself_loss)
            precon_loss = pself_loss = None
        
        print(out)
        
        return paramses, pvecs, mloss
    
    def train_slave_finetune(self, task_batch_size, query_loaders, param, pvec, mloss, epoch, total, step, k, sub = ''):
        
        loss_fs = []
        mgraph_embses = []
        msubgraph_embses = []
        mmi_losses = []
        
        maccs = []
        maucs = []
        mf1s = []
        mpre_losses = []
        #mrecon_losses = []
        #mself_losses = []
        mpenalties = []
        
        for t in range(task_batch_size):
            tname, query_loader = query_loaders[t]
            #vector_to_parameters(old_params, self.slearner.parameters())
            #self.load_params((param[0][t].squeeze(), param[1][t].squeeze()))
            
            macc, mauc, mf1, mpre_loss, _, _, mppenalties, mgraph_embs, msubgraph_embs = self.train_slave_((param[0][t].squeeze(), param[1][t].squeeze()), query_loader)
            
            #self.load_params((param[0][t].squeeze(), param[1][t].squeeze()))
            #macc, mauc, mf1, mpre_loss, mrecon_loss, mself_loss, mppenalties, mgraph_embs, msubgraph_embs = self.train_slave(query_loader)
            
            mgraph_embses.append(mgraph_embs)
            msubgraph_embses.append(msubgraph_embs)
            
            maccs.append(macc)
            maucs.append(mauc)
            mf1s.append(mf1)
            
            mpre_losses.append(mpre_loss)
            #mrecon_losses.append(mrecon_loss)
            #mself_losses.append(mself_loss)
            mpenalties.append(mppenalties)
            
            out = 'EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|'+str(t)+'.'+tname+':\t'
            out += 'sub finetune loss:'+str(round(mpre_loss.item(),2))
            #out += '|reconl:'+str(round(mrecon_loss.item(),2))
            #if self.add_selfsupervise:
            #    out += '|selfl:'+str(round(mself_loss.item(),2))
            out += '|ppenalty:'+str(round(mppenalties.item(),2))
            print(out)
            
            loss_f = (mpre_loss + self.ex_weight * (mppenalties))# + mself_loss + mrecon_loss))
            loss_fs.append(loss_f.unsqueeze(dim = 0))
            
            macc = mauc = mf1 = mpre_loss = _ = mppenalties = mgraph_embs = msubgraph_embs = loss_f = None
        
        loss_fs = torch.cat(loss_fs).squeeze()
        
        flocal_losses = []
        for _ in range(self.inner_loop):
            local_loss = 0
            optimizer_local = torch.optim.Adam(self.f_disc.parameters(),lr=self.meta_lr, weight_decay=self.decay)
            optimizer_local.zero_grad()
            for i in range(len(mgraph_embses)):
                local_loss += ((0.1*self.ex_weight) if self.fdisc_flag else 1.0) * (- MI_Est(self.f_disc, mgraph_embses[i].clone().detach_(), msubgraph_embses[i].clone().detach_()))
            local_loss = local_loss/len(mgraph_embses)
            local_loss.backward()#retain_graph = True)
            optimizer_local.step()
            flocal_losses.append(local_loss.clone().detach_().item())
        self.f_disc.save_params()
        for i in range(len(mgraph_embses)):
            mmi_losses.append(MI_Est(self.f_disc, mgraph_embses[i], msubgraph_embses[i]).unsqueeze(dim = 0))
        mmi_losses = torch.cat(mmi_losses).squeeze()
        print('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'| '+sub+'finetune MI_Est loss:'+str(mmi_losses.clone().detach_().cpu()))
        loss_fs += self.ex_weight * beta * mmi_losses.squeeze()
        self.fdisc_flag = self.f_disc.tune_disc(torch.mean(mmi_losses), self.fdisc_flag) if self.tune_disc else self.fdisc_flag
        
        mgraph_embses = msubgraph_embses = None
        
        self.writer.add_scalar('00.Train finetune '+sub+'accs',sum(maccs)/len(maccs),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train finetune '+sub+'aucs',sum(maucs)/len(maucs),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train finetune '+sub+'f1s',sum(mf1s)/len(maccs),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train finetune '+sub+'pred loss',sum(mpre_losses)/len(mpre_losses),epoch*total*self.update_step+step*self.update_step+k)
        #self.writer.add_scalar('00.Train finetune '+sub+'recon loss',sum(mrecon_losses)/len(mrecon_losses),epoch*total*self.update_step+step*self.update_step+k)
        #self.writer.add_scalar('00.Train finetune '+sub+'self loss',sum(mself_losses)/len(mself_losses),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train finetune '+sub+'penalties',sum(mpenalties)/len(mpenalties),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train finetune '+sub+'mi loss',sum(mmi_losses)/len(mmi_losses),epoch*total*self.update_step+step*self.update_step+k)
        self.writer.add_scalar('00.Train finetune '+sub+'disc loss',sum(flocal_losses)/len(flocal_losses),epoch*total*self.update_step+step*self.update_step+k)
        
        if self.add_metasimilarity:
            if self.attention_detach:
                loss_m = torch.sum(self.meta_attention(pvec.detach()) * (loss_fs + mloss))
            else:
                loss_m = torch.sum(self.meta_attention(pvec) * (loss_fs + mloss))
        else:
            loss_m = torch.sum(loss_fs + mloss)/task_batch_size
        print('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|'+sub+'Finetune Loss:'+str(loss_m))
        self.writer.add_scalar('00.Train finetune '+sub+'loss', loss_m, epoch*total*self.update_step+step*self.update_step+k)
        
        return loss_m
    
    def forward(self, task_batch, epoch, total, step):
        
        self.m_disc.save_params()
        self.s_disc.save_params()
        self.f_disc.save_params()
        
        task_batch_size = len(task_batch)
        protein_dataset = []
        protein_d = []
        support_loaders = []
        support_l = []
        query_loaders = []
        query_l = []
        index = []
        for tb in range(task_batch_size):
            (tname, dataset) = task_batch[tb]
            protein_d.append((tname,)+dataset.get_protein())
            support_dataset, query_dataset = sample_datasets(dataset.get_ligands(), tname, self.n_way, self.m_support, self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False)
            support_l.append((tname,support_loader))
            query_l.append((tname,query_loader))
            index.append(tb)
        
        random.shuffle(index)
        
        for i in index:
            protein_dataset.append(protein_d[i])
            support_loaders.append(support_l[i])
            query_loaders.append(query_l[i])
            assert support_l[i][0] == query_l[i][0]
        
        self.slearner.train()
        print('\ntraining slave learner ......')
        
        old_params = parameters_to_vector(self.slearner.parameters()).clone().detach_()
        train_losses = []
        for k in range(self.update_step):
            
            loss_q = self.run_slave(task_batch_size, support_loaders, query_loaders, old_params, epoch, total, step, k)
            
            if not self.slave_fix:
                old_params = parameters_to_vector(self.slearner.parameters()).clone().detach_()
            if self.mode == 1:
                train_losses.append(loss_q.clone().detach_().cpu().item())
            
            if self.mode == 0 or (self.mode == 1 and k == self.update_step - 1):
                slearner_params1, slearner_params2 = self.run_slave_finetune(task_batch_size, query_loaders, epoch, total, step, k)
                    
            if self.mode == 0:
                paramses, pvecs, mloss = self.train_master_finetune(slearner_params1, slearner_params2, protein_dataset, epoch, total, step, k)
                
                #==============================train slave finetune===================================
                '''
                #train for global_graph_pvec
                loss_m = self.train_slave_finetune(task_batch_size, query_loaders, paramses[0], pvecs[0], mloss, epoch, total, step, k, sub = '')
                '''
                #train for sub_graph_pvec 
                
                loss_m_s = self.train_slave_finetune(task_batch_size, query_loaders, paramses[1], pvecs[1], mloss, epoch, total, step, k, sub = 'sub ')
                
                finetune_loss = loss_m_s# + loss_m
                
                #finetune_loss += self.ex_weight * pmi_loss
                
                print('EPOCH:'+str(epoch)+'|'+'Step:'+str(step)+'|'+'UpdateStep:'+str(k)+'|Total Loss:'+str(finetune_loss))
                print()
                self.writer.add_scalar('00.Train finetune total loss', finetune_loss,epoch*total*self.update_step+step*self.update_step+k)
                
                self.optimizer_.zero_grad()
                finetune_loss.backward()
                self.optimizer_.step()
                
                train_losses.append(finetune_loss.clone().detach_().cpu().item())
                
                finetune_loss = loss_m_s = pvecs = slearner_params1 = slearner_params2 = paramses = mloss = None
                
                vector_to_parameters(old_params.clone(), self.slearner.parameters())
                
        if self.mode == 1:
            self.mlearner(protein_dataset, (slearner_params1, slearner_params2))
            return sum(train_losses)/len(train_losses)
        elif self.mode == 0:
            return sum(train_losses)/len(train_losses)
        else:
            raise ValueError('unimplementd train mode')
    
    def test(self, task_batch, epoch, anchor = '', extra = False):
        if extra:
            print('provide prec and recall')
        accs = []
        roc_list = []
        f1_list = []
        accs0 = []
        roc_list0 = []
        f1_list0 = []
        if extra:
            prec_list = []
            prec_list0 = []
            recall_list = []
            recall_list0 = []
        
        if type(epoch) != str:
            pred_losses = []
            pred0_losses = []
            recon_losses = []
            recon0_losses = []
            penalty_losses = []
            penalty0_losses = []
            self_losses = []
            self0_losses = []
            losses0 = []
            losses = []
        
        with torch.no_grad():
            old_params = parameters_to_vector(self.slearner.parameters()).clone().detach_()
        
        for (tname, dataset) in task_batch:
            s_disc = copy_disc(self.s_disc).cuda()
            s_disc.reset_tunecounter()
            test_sdisc_flag = False
            
            pdata, _ = dataset.get_protein()
            pdata = pdata.cuda()
            support_dataset, query_dataset = sample_test_datasets(dataset.get_ligands(), tname, self.n_way, self.m_support, self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False)
            
            self.slearner.eval()
            self.mlearner.eval()
            
            with torch.no_grad():
                t_params = self.mlearner.test(pdata)
                t_params = (t_params[0].detach(), t_params[1].detach())
            self.load_params(t_params)
            t_params = None
            
            with torch.no_grad():
                if extra:
                    acc, auc, f1, prec, recall, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(query_loader, extra)
                else:
                    acc, auc, f1, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(query_loader, extra)
                accs0.append(acc)
                roc_list0.append(auc)
                f1_list0.append(f1)
                if extra:
                    prec_list0.append(prec)
                    recall_list0.append(recall)
                print('EPOCH '+str(epoch)+'|'+tname+'\tstart acc:'+str(round(acc,2))+'|start auc:'+str(round(auc,2))+'|start f1:'+str(round(f1,2)))
                if type(epoch) != str:
                    pred0_losses.append(pre_loss)
                    recon0_losses.append(recon_loss)
                    penalty0_losses.append(penalties)
                    self0_losses.append(self_loss)
                    losses0.append(pre_loss + self.ex_weight * (recon_loss + penalties + self_loss))
            
            test_optimizer = optim.RMSprop(self.slearner.parameters(), lr = self.test_update_lr, weight_decay=self.decay)
            s_disc.save_params()
            for k in range(self.update_step_test):
                loss = 0
                acc, auc, f1, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(support_loader)
                
                for _ in range(self.inner_loop):
                    optimizer_local = torch.optim.Adam(s_disc.parameters(),lr=self.meta_lr, weight_decay=self.decay)
                    optimizer_local.zero_grad()
                    local_loss = ((0.1*self.ex_weight) if test_sdisc_flag else 1.0) * - MI_Est(s_disc, graph_embs.clone().detach_(), subgraph_embs.clone().detach_())
                    local_loss.backward()#retain_graph = True)
                    optimizer_local.step()
                mi_loss = MI_Est(s_disc, graph_embs, subgraph_embs)
                test_sdisc_flag = s_disc.tune_disc(mi_loss, test_sdisc_flag) if self.tune_disc else test_sdisc_flag
                
                loss += (pre_loss + self.ex_weight * (recon_loss + self_loss + penalties + beta * mi_loss))
                test_optimizer.zero_grad()
                loss.backward()
                test_optimizer.step()
                
                s_disc.reset_params()
                
            with torch.no_grad():
                if extra:
                    acc, auc, f1, prec, recall, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(query_loader, extra)
                else:
                    acc, auc, f1, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(query_loader, extra)
                accs.append(acc)
                roc_list.append(auc)
                f1_list.append(f1)
                if extra:
                    prec_list0.append(prec)
                    recall_list0.append(recall)
                print('EPOCH '+str(epoch)+'|'+tname+'\tfinetune acc:'+str(round(acc,2))+'|finetune auc:'+str(round(auc,2))+'|finetune f1:'+str(round(f1,2)))
                if type(epoch) != str:
                    pred_losses.append(pre_loss)
                    recon_losses.append(recon_loss)
                    penalty_losses.append(penalties)
                    self_losses.append(self_loss)
                    losses.append(pre_loss + self.ex_weight * (recon_loss + penalties + self_loss))
                
            vector_to_parameters(old_params.clone().detach_(), self.slearner.parameters())
            test_sdisc_flag = False
        
        if type(epoch) != str:
            self.writer.add_scalar('00.Test start'+anchor+' acc', sum(accs0)/len(accs0), epoch)
            self.writer.add_scalar('00.Test'+anchor+' acc', sum(accs)/len(accs0), epoch)
            self.writer.add_scalar('00.Test start'+anchor+' auc', sum(roc_list0)/len(roc_list0), epoch)
            self.writer.add_scalar('00.Test'+anchor+' auc', sum(roc_list)/len(roc_list), epoch)
            self.writer.add_scalar('00.Test start'+anchor+' f1', sum(f1_list0)/len(f1_list0), epoch)
            self.writer.add_scalar('00.Test'+anchor+' f1', sum(f1_list)/len(f1_list), epoch)
            self.writer.add_scalar('00.Test start'+anchor+' pred loss', sum(pred0_losses)/len(pred0_losses), epoch)
            self.writer.add_scalar('00.Test'+anchor+' pred loss', sum(pred_losses)/len(pred_losses), epoch)
            self.writer.add_scalar('00.Test start'+anchor+' recon loss', sum(recon0_losses)/len(recon0_losses), epoch)
            self.writer.add_scalar('00.Test'+anchor+' recon loss', sum(recon_losses)/len(recon_losses), epoch)
            self.writer.add_scalar('00.Test start'+anchor+' penalties', sum(penalty0_losses)/len(penalty0_losses), epoch)
            self.writer.add_scalar('00.Test'+anchor+' penalties', sum(penalty_losses)/len(penalty_losses), epoch)
            if self.add_selfsupervise:
                self.writer.add_scalar('00.Test start self loss', sum(self0_losses)/len(self0_losses), epoch)
                self.writer.add_scalar('00.Test self loss', sum(self_losses)/len(self_losses), epoch)
            self.writer.add_scalar('00.Test start loss', sum(losses0)/len(losses0), epoch)
            self.writer.add_scalar('00.Test loss', sum(losses)/len(losses), epoch)
        if extra:
            return accs0, roc_list0, accs, roc_list, f1_list0, f1_list, prec_list0, prec_list, recall_list0, recall_list
        return accs0, roc_list0, accs, roc_list, f1_list0, f1_list
    
    def test_(self, task_batch, epoch):
        accs = []
        roc_list = []
        f1_list = []
        accs0 = []
        roc_list0 = []
        f1_list0 = []
        
        if type(epoch) != str:
            pred_losses = []
            pred0_losses = []
            recon_losses = []
            recon0_losses = []
            penalty_losses = []
            penalty0_losses = []
            self_losses = []
            self0_losses = []
            losses0 = []
            losses = []
        
        with torch.no_grad():
            old_params = parameters_to_vector(self.slearner.parameters()).clone().detach_()
        
        for (tname, dataset) in task_batch:
            s_disc = copy_disc(self.s_disc).cuda()
            s_disc.reset_tunecounter()
            test_sdisc_flag = False
            #pdata, _ = dataset.get_protein()
            #pdata = pdata.cuda()
            support_dataset, query_dataset = sample_test_datasets(dataset.get_ligands(), tname, self.n_way, self.m_support, self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False)
            
            self.slearner.eval()
            #self.mlearner.eval()
            
            #with torch.no_grad():
            #    t_params = self.mlearner.test(pdata)
            #    t_params = (t_params[0].detach(), t_params[1].detach())
            #self.load_params(t_params)
            #t_params = None
            
            with torch.no_grad():
                acc, auc, f1, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(query_loader)
                accs0.append(acc)
                roc_list0.append(auc)
                f1_list0.append(f1)
                print('EPOCH '+str(epoch)+'|'+tname+'\tstart acc:'+str(round(acc,2))+'|start auc:'+str(round(auc,2))+'|start f1:'+str(round(f1,2)))
                if type(epoch) != str:
                    pred0_losses.append(pre_loss)
                    recon0_losses.append(recon_loss)
                    penalty0_losses.append(penalties)
                    self0_losses.append(self_loss)
                    losses0.append(pre_loss + self.ex_weight * (recon_loss + penalties + self_loss))
            
            test_optimizer = optim.RMSprop(self.slearner.parameters(), lr = self.test_update_lr, weight_decay=self.decay)
            s_disc.save_params()
            for k in range(self.update_step_test):
                loss = 0
                acc, auc, f1, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(support_loader)
                
                for _ in range(self.inner_loop):
                    optimizer_local = torch.optim.Adam(s_disc.parameters(),lr=self.meta_lr, weight_decay=self.decay)
                    optimizer_local.zero_grad()
                    local_loss = ((0.1*self.ex_weight) if test_sdisc_flag else 1.0) * - MI_Est(s_disc, graph_embs.clone().detach_(), subgraph_embs.clone().detach_())
                    local_loss.backward()#retain_graph = True)
                    optimizer_local.step()
                mi_loss = MI_Est(s_disc, graph_embs, subgraph_embs)
                test_sdisc_flag = s_disc.tune_disc(mi_loss, test_sdisc_flag) if self.tune_disc else test_sdisc_flag
                
                loss += (pre_loss + self.ex_weight * (recon_loss + self_loss + penalties + beta * mi_loss))
                test_optimizer.zero_grad()
                loss.backward()
                test_optimizer.step()
                
                s_disc.reset_params()
                
            with torch.no_grad():
                acc, auc, f1, pre_loss, recon_loss, self_loss, penalties, graph_embs, subgraph_embs = self.train_slave(query_loader)
                accs.append(acc)
                roc_list.append(auc)
                f1_list.append(f1)
                print('EPOCH '+str(epoch)+'|'+tname+'\tfinetune acc:'+str(round(acc,2))+'|finetune auc:'+str(round(auc,2))+'|finetune f1:'+str(round(f1,2)))
                if type(epoch) != str:
                    pred_losses.append(pre_loss)
                    recon_losses.append(recon_loss)
                    penalty_losses.append(penalties)
                    self_losses.append(self_loss)
                    losses.append(pre_loss + self.ex_weight * (recon_loss + penalties + self_loss))
                
            vector_to_parameters(old_params.clone().detach_(), self.slearner.parameters())
            test_sdisc_flag=False
        
        if type(epoch) != str:
            self.writer.add_scalar('00.Test start acc', sum(accs0)/len(accs0), epoch)
            self.writer.add_scalar('00.Test acc', sum(accs)/len(accs0), epoch)
            self.writer.add_scalar('00.Test start auc', sum(roc_list0)/len(roc_list0), epoch)
            self.writer.add_scalar('00.Test auc', sum(roc_list)/len(roc_list), epoch)
            self.writer.add_scalar('00.Test start f1', sum(f1_list0)/len(f1_list0), epoch)
            self.writer.add_scalar('00.Test f1', sum(f1_list)/len(f1_list), epoch)
            self.writer.add_scalar('00.Test start pred loss', sum(pred0_losses)/len(pred0_losses), epoch)
            self.writer.add_scalar('00.Test pred loss', sum(pred_losses)/len(pred_losses), epoch)
            self.writer.add_scalar('00.Test start recon loss', sum(recon0_losses)/len(recon0_losses), epoch)
            self.writer.add_scalar('00.Test recon loss', sum(recon_losses)/len(recon_losses), epoch)
            self.writer.add_scalar('00.Test start penalties', sum(penalty0_losses)/len(penalty0_losses), epoch)
            self.writer.add_scalar('00.Test penalties', sum(penalty_losses)/len(penalty_losses), epoch)
            if self.add_selfsupervise:
                self.writer.add_scalar('00.Test start self loss', sum(self0_losses)/len(self0_losses), epoch)
                self.writer.add_scalar('00.Test self loss', sum(self_losses)/len(self_losses), epoch)
            self.writer.add_scalar('00.Test start loss', sum(losses0)/len(losses0), epoch)
            self.writer.add_scalar('00.Test loss', sum(losses)/len(losses), epoch)
        
        return accs0, roc_list0, accs, roc_list, f1_list0, f1_list
                
    
