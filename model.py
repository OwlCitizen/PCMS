#model.py
import torch
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, remove_self_loops, to_dense_adj, add_self_loops
from torch.nn import ModuleList, ModuleDict, Linear, Sequential
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from typing import Optional
from collections import defaultdict
from torch_geometric.nn.conv.hgt_conv import group
from torch_geometric.nn import HeteroConv, TransformerConv, GATConv, radius_graph
from torch_scatter import scatter

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3
EPS = 1e-15

class SchNet(torch.nn.Module):
    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32, drop_ratio = 0.5,
                 readout: str = 'add', dipole: bool = False, pred : bool = True,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        super(SchNet, self).__init__()
        
        #import ase

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        #atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        #self.register_buffer('atomic_mass', atomic_mass)

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_channels)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        #self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        #self.act = ShiftedSoftplus()
        #self.lin2 = Linear(hidden_channels // 2, 1)

        #self.register_buffer('initial_atomref', atomref)
        #self.atomref = None
        #if atomref is not None:
        #    self.atomref = Embedding(100, 1)
        #    self.atomref.weight.data.copy_(atomref)
        
        if self.pred:
            self.graph_pred_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_channels, self.hidden_channels//2),
                    #torch.nn.LeakyReLU(inplace = True),
                    ShiftedSoftplus(),
                    torch.nn.Dropout(drop_ratio),
                    torch.nn.Linear(self.hidden_channels//2,2),
                )
        
        self.reset_parameters()

    def reset_parameters(self):
        self.x_embedding1.reset_parameters()
        self.x_embedding2.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        #torch.nn.init.xavier_uniform_(self.lin1.weight)
        #self.lin1.bias.data.fill_(0)
        #torch.nn.init.xavier_uniform_(self.lin2.weight)
        #self.lin2.bias.data.fill_(0)
        
        #if self.atomref is not None:
        #    self.atomref.weight.data.copy_(self.initial_atomref)
        if self.pred:
            torch.nn.init.xavier_uniform_(self.graph_pred_linear[0].weight)
            self.graph_pred_linear[0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.graph_pred_linear[3].weight)
            self.graph_pred_linear[3].bias.data.fill_(0)

    def forward(self, h, edge_index, edge_attr, batch=None):
        """"""
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch

        h = self.x_embedding1(h[:,0].long()) + self.x_embedding1(h[:,1].long())
        pos = h[:,2:]
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        #edge_weight = edge_attr[:,:3].norm(dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        for l, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            h = self.batch_norms[l](h)
        
        #h = self.lin1(h)
        #h = self.act(h)
        #h = self.lin2(h)
        
        out = scatter(h, batch, dim=0, reduce=self.readout)
        
        if self.pred:
            return self.graph_pred_linear(out), out, h
        else:
            return out, h

class SchNet_(torch.nn.Module):
    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        super(SchNet_, self).__init__()

        #import ase

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        #atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        #self.register_buffer('atomic_mass', atomic_mass)

        #self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_channels)
        #self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        #self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        #self.act = ShiftedSoftplus()
        #self.lin2 = Linear(hidden_channels // 2, 1)

        #self.register_buffer('initial_atomref', atomref)
        #self.atomref = None
        #if atomref is not None:
        #    self.atomref = Embedding(100, 1)
        #    self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        #self.x_embedding1.reset_parameters()
        #self.x_embedding2.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        
        #torch.nn.init.xavier_uniform_(self.lin1.weight)
        #self.lin1.bias.data.fill_(0)
        #torch.nn.init.xavier_uniform_(self.lin2.weight)
        #self.lin2.bias.data.fill_(0)
        
        #if self.atomref is not None:
        #    self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, h, pos, edge_index, edge_attr, batch=None):
        """"""
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch

        #h = self.x_embedding1(h[:,0].long()) + self.x_embedding1(h[:,1].long())
        #pos = h[:,2:]
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        #edge_weight = edge_attr[:,:3].norm(dim = -1)
        edge_attr = self.distance_expansion(edge_weight)
        #print('edge attr in SchNet_：')
        #print(edge_attr)
        #print(edge_attr.dtype)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        #h = self.lin1(h)
        #h = self.act(h)
        #h = self.lin2(h)

        return h, edge_index

class MySchNet(torch.nn.Module):
    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32, drop_ratio = 0.5,
                 readout: str = 'add', dipole: bool = False, pred : bool = True,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        super(MySchNet, self).__init__()
        
        #import ase

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.pred = pred

        #atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        #self.register_buffer('atomic_mass', atomic_mass)

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_channels)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        #self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        #self.act = ShiftedSoftplus()
        #self.lin2 = Linear(hidden_channels // 2, 1)

        #self.register_buffer('initial_atomref', atomref)
        #self.atomref = None
        #if atomref is not None:
        #    self.atomref = Embedding(100, 1)
        #    self.atomref.weight.data.copy_(atomref)
        
        if self.pred:
            self.graph_pred_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_channels, self.hidden_channels//2),
                    torch.nn.LeakyReLU(inplace = True),
                    torch.nn.Dropout(drop_ratio),
                    torch.nn.Linear(self.hidden_channels//2,2),
                )
        
        self.reset_parameters()

    def reset_parameters(self):
        self.x_embedding1.reset_parameters()
        self.x_embedding2.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
            
        #torch.nn.init.xavier_uniform_(self.lin1.weight)
        #self.lin1.bias.data.fill_(0)
        #torch.nn.init.xavier_uniform_(self.lin2.weight)
        #self.lin2.bias.data.fill_(0)
        
        #if self.atomref is not None:
        #    self.atomref.weight.data.copy_(self.initial_atomref)
        if self.pred:
            torch.nn.init.xavier_uniform_(self.graph_pred_linear[0].weight)
            self.graph_pred_linear[0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.graph_pred_linear[3].weight)
            self.graph_pred_linear[3].bias.data.fill_(0)

    def forward(self, h, edge_index, edge_attr, batch=None):
        """"""
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch
        
        h = self.x_embedding1(h[:,0].long()) + self.x_embedding2(h[:,1].long())
        
        row, col = edge_index
        #edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_weight = edge_attr[:,:3].norm(dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        for l, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            h = self.batch_norms[l](h)
        
        #h = self.lin1(h)
        #h = self.act(h)
        #h = self.lin2(h)
        
        out = scatter(h, batch, dim=0, reduce=self.readout)
        
        if self.pred:
            return self.graph_pred_linear(out), out, h
        else:
            return out, h
    
    def graph_pred_linear_(self, params, h):
        weight1_shape = self.graph_pred_linear[0].weight.shape
        bias1_shape = self.graph_pred_linear[0].bias.shape
        weight2_shape = self.graph_pred_linear[2].weight.shape
        bias2_shape = self.graph_pred_linear[2].bias.shape
        
        start = 0
        weight1 = params[start:start+weight1_shape[0]*weight1_shape[1]].view(weight1_shape).t()
        start += weight1_shape[0]*weight1_shape[1]
        bias1 = params[start:start+bias1_shape[-1]].view(bias1_shape)
        start += bias1_shape[0]
        weight2 = params[start:start+weight2_shape[0]*weight2_shape[1]].view(weight2_shape).t()
        start += weight2_shape[0]*weight2_shape[1]
        bias2 = params[start:start+bias2_shape[-1]].view(bias2_shape)
        start += bias2_shape[0]
        assert start == len(params)
        
        out = self.graph_pred_linear[1](torch.matmul(h, weight1)+bias1)
        out = torch.matmul(out, weight2)+bias2
        return out
    
    def forward_(self, params, h, edge_index, edge_attr, batch=None):
        """"""
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch
        
        h = self.x_embedding1(h[:,0].long()) + self.x_embedding2(h[:,1].long())
        
        row, col = edge_index
        #edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_weight = edge_attr[:,:3].norm(dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        for l, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            h = self.batch_norms[l](h)
        
        #h = self.lin1(h)
        #h = self.act(h)
        #h = self.lin2(h)
        
        out = scatter(h, batch, dim=0, reduce=self.readout)
        
        if self.pred:
            pred = self.graph_pred_linear_(params, out)
            return pred, out, h
        else:
            return out, h

class MySchNet_(torch.nn.Module):
    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        super(MySchNet_, self).__init__()

        #import ase

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        #atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        #self.register_buffer('atomic_mass', atomic_mass)

        #self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_channels)
        #self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        #self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        #self.act = ShiftedSoftplus()
        #self.lin2 = Linear(hidden_channels // 2, 1)

        #self.register_buffer('initial_atomref', atomref)
        #self.atomref = None
        #if atomref is not None:
        #    self.atomref = Embedding(100, 1)
        #    self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        #self.x_embedding1.reset_parameters()
        #self.x_embedding2.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        
        #torch.nn.init.xavier_uniform_(self.lin1.weight)
        #self.lin1.bias.data.fill_(0)
        #torch.nn.init.xavier_uniform_(self.lin2.weight)
        #self.lin2.bias.data.fill_(0)
        
        #if self.atomref is not None:
        #    self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, h, edge_index, edge_attr, batch=None):
        """"""
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch

        #h = self.x_embedding1(h[:,0].long()) + self.x_embedding1(h[:,1].long())
        
        row, col = edge_index
        #edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_weight = edge_attr[:,:3].norm(dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        #h = self.lin1(h)
        #h = self.act(h)
        #h = self.lin2(h)

        return h

class MySchNet__(torch.nn.Module):
    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        super(MySchNet__, self).__init__()

        #import ase

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        #atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        #self.register_buffer('atomic_mass', atomic_mass)

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_channels)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        #self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        #self.act = ShiftedSoftplus()
        #self.lin2 = Linear(hidden_channels // 2, 1)

        #self.register_buffer('initial_atomref', atomref)
        #self.atomref = None
        #if atomref is not None:
        #    self.atomref = Embedding(100, 1)
        #    self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.x_embedding1.reset_parameters()
        self.x_embedding2.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        
        #torch.nn.init.xavier_uniform_(self.lin1.weight)
        #self.lin1.bias.data.fill_(0)
        #torch.nn.init.xavier_uniform_(self.lin2.weight)
        #self.lin2.bias.data.fill_(0)
        
        #if self.atomref is not None:
        #    self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, h, edge_index, edge_attr, batch=None):
        """"""
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch

        h = self.x_embedding1(h[:,0].long()) + self.x_embedding1(h[:,1].long())
        
        row, col = edge_index
        #edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_weight = edge_attr[:,:3].norm(dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        #h = self.lin1(h)
        #h = self.act(h)
        #h = self.lin2(h)

        return h

class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn_, cutoff):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn_ = nn_
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        W = self.nn_(edge_attr.float()) * C.view(-1, 1)
        
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x.float())
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class ProteinHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels1=128, hidden_channels2 = 1024, num_layers=2):#, batchnorm = False):
        super().__init__()
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2
        
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_channels1)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_channels1)
        self.convs = torch.nn.ModuleList()
        #self.batchnorm = batchnorm
        #if self.batchnorm:
        #    self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('atom', 'bond', 'atom'): MySchNet_(hidden_channels = hidden_channels1, num_interactions=3),
                ('atom', 'belong', 'residue'): GATConv((hidden_channels1, hidden_channels2), hidden_channels2),
                ('residue', 'of', 'atom'): GATConv((hidden_channels2, hidden_channels1), hidden_channels1),
                ('residue', 'nextto', 'residue'): TransformerConv(hidden_channels2, hidden_channels2)
            }, aggr='sum')
            self.convs.append(conv)
            #if self.batchnorm:
            #    bn = ModuleDict({'atom': torch.nn.BatchNorm1d(hidden_channels1),'residue': torch.nn.BatchNorm1d(hidden_channels2)})
            #    self.bns.append(bn)
            
        self.lin1 = Linear(hidden_channels1, hidden_channels1)
        self.lin2 = Linear(hidden_channels2, hidden_channels1)
        
        self.x_embedding1.reset_parameters()
        self.x_embedding1.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        pos = x_dict['atom'][:,2:]
        x_dict['atom'] = self.x_embedding1(x_dict['atom'][:,0].long()) + self.x_embedding2(x_dict['atom'][:,1].long())
        for layer, conv in enumerate(self.convs):
            model_dict = conv.convs
            out_dict = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                str_edge_type = '__'.join(edge_type)
                #print(str_edge_type)
                model = model_dict[str_edge_type]
                if str_edge_type == 'atom__bond__atom':
                    out = model(x_dict[src], edge_index, edge_attr_dict[('atom','bond','atom')])
                elif str_edge_type == 'atom__belong__residue':
                    out = model((x_dict['atom'], x_dict['residue']), edge_index)
                elif str_edge_type == 'residue__of__atom':
                    out = model((x_dict['residue'], x_dict['atom']), edge_index)
                elif str_edge_type == 'residue__nextto__residue':
                    out = model(x_dict['residue'], edge_index)
                else:
                    raise ValueError('unimplemented heterogeneous graph edge!')
                out_dict[dst].append(out)
                
            for key, value in out_dict.items():
                x_dict[key] = group(value, 'sum')
                
            #x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        emb_atoms, emb_residues = self.lin1(x_dict['atom']), self.lin2(x_dict['residue'])
        
        return emb_atoms, edge_index_dict[('atom','bond','atom')], emb_residues, torch.cat([torch.sum(emb_atoms, dim = 0).view(1, self.hidden_channels1), torch.sum(emb_residues, dim=0).view(1, self.hidden_channels1)],dim = 1)

class GIB(nn.Module):
    def __init__(self, num_emb, drop_ratio = 0.5, con_weight = 5, connect = False, assignment = False, pred = False):
        super(GIB, self).__init__()
        self.num_emb = num_emb
        self.con_weight = con_weight
        self.pred = pred
        self.connect = connect
        self.assignment = assignment
        
        self.graph_assignment = nn.Sequential(
                nn.Linear(self.num_emb, self.num_emb // 2),
                nn.Tanh(),
                nn.Linear(self.num_emb // 2, 2),
                nn.Softmax(dim = 1)
            )
        
        self.mseloss = torch.nn.MSELoss()
        
        if self.pred:
            self.graph_pred_linear = torch.nn.Sequential(
                   torch.nn.Linear(self.num_emb, self.num_emb // 2),
                   #torch.nn.LeakyReLU(inplace = True),
                   ShiftedSoftplus(),
                   #torch.nn.Dropout(drop_ratio),
                   torch.nn.Linear(self.num_emb // 2,2),
               )
        
    def forward(self, h, edge_index, batch=None):
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch
        
        assignment_all = self.graph_assignment(h)
        
        graph_embeddings = []
        if self.assignment:
            assignments = []
        positives = []
        #negatives = []
        positive_penalties = []
        start = 0
        for b in range(batch.max()+1):
            #edge_start = []
            #edge_end = []
            node_index = (batch == b).nonzero().squeeze()
            
            assignment = assignment_all[node_index]
            group_features = torch.mm(torch.t(assignment), h[node_index])
            if self.assignment:
                assignments.append(assignment[0].unsqueeze(dim = 1))
            
            #positive = torch.clamp(group_features[0].unsqueeze(dim = 0),-100,100)
            positive = group_features[0].unsqueeze(dim = 0)
            #negative = torch.clamp(group_features[1].unsqueeze(dim = 0),-100,100)
            
            graph_embedding = torch.mm(torch.t(assignment), h[node_index])
            graph_embedding = torch.mean(graph_embedding,dim = 0, keepdim= True)
            
            graph_embeddings.append(graph_embedding.view(1, self.num_emb))
            positives.append(positive.view(1, self.num_emb))
            #negatives.append(negative.view(1, self.num_emb))
            
            if self.connect:
                if h.is_cuda:
                    EYE = torch.ones(2).cuda()
                else:
                    EYE = torch.ones(2)
                for j in range(start, edge_index.shape[1]):
                    if edge_index[0][j] not in node_index or edge_index[1][j] not in node_index:
                        sub_edge_index_o = edge_index[:,start:j]
                        start = j
                        break
                    elif j == edge_index.shape[1]-1:
                        sub_edge_index_o = edge_index[:,start:]
                #sub_edge_index = torch.cat([torch.tensor(edge_start).unsqueeze(dim=0),torch.tensor(edge_end).unsqueeze(dim=0)],dim = 0)
                #sub_edge_index_o = sub_edge_index
                if b == batch.max() and j < edge_index.shape[1]-1:
                    print(node_index)
                    print(edge_index[:,j:])
                    print(edge_index)
                    raise ValueError('remain batch edge index in gib!')
                sub_edge_index = sub_edge_index_o - sub_edge_index_o.min()
                #sub_edge_index = sub_edge_index.cuda() if h.is_cuda else sub_edge_index
                Adj = to_dense_adj(sub_edge_index, max_num_nodes=len(node_index))[0]
                Adj.requires_grad = False
                new_adj = torch.mm(torch.t(assignment),Adj)
                new_adj = torch.mm(new_adj,assignment)
                
                normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
                norm_diag = torch.diag(normalize_new_adj)
                pos_penalty = self.mseloss(norm_diag, EYE)
                
                positive_penalties.append(pos_penalty.view(1))
                
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        positives = torch.cat(positives, dim=0)
        if self.pred:
            pred = self.graph_pred_linear(torch.cat([graph_embeddings, positives], dim=0))
            if self.assignment:
                return pred, h, graph_embeddings, positives, assignments, positive_penalties
            else:
                return pred, h, graph_embeddings, positives, positive_penalties
        if self.assignment:
            return h, graph_embeddings, positives, assignments, positive_penalties
        else:
            return h, graph_embeddings, positives, positive_penalties
    
    def graph_assignment_(self, params, h):
        weight1_shape = self.graph_assignment[0].weight.shape
        bias1_shape = self.graph_assignment[0].bias.shape
        weight2_shape = self.graph_assignment[2].weight.shape
        bias2_shape = self.graph_assignment[2].bias.shape
        
        start = 0
        weight1 = params[start:start+weight1_shape[0]*weight1_shape[1]].view(weight1_shape).t()
        start += weight1_shape[0]*weight1_shape[1]
        bias1 = params[start:start+bias1_shape[-1]].view(bias1_shape)
        start += bias1_shape[0]
        weight2 = params[start:start+weight2_shape[0]*weight2_shape[1]].view(weight2_shape).t()
        start += weight2_shape[0]*weight2_shape[1]
        bias2 = params[start:start+bias2_shape[-1]].view(bias2_shape)
        start += bias2_shape[0]
        assert start == len(params)
        
        out = torch.tanh(torch.matmul(h, weight1)+bias1)
        out = torch.softmax((torch.matmul(out, weight2)+bias2), dim=1)
        return out
    
    def graph_pred_linear_(self, params, h):
        weight1_shape = self.graph_pred_linear[0].weight.shape
        bias1_shape = self.graph_pred_linear[0].bias.shape
        weight2_shape = self.graph_pred_linear[2].weight.shape
        bias2_shape = self.graph_pred_linear[2].bias.shape
        
        start = 0
        weight1 = params[start:start+weight1_shape[0]*weight1_shape[1]].view(weight1_shape).t()
        start += weight1_shape[0]*weight1_shape[1]
        bias1 = params[start:start+bias1_shape[-1]].view(bias1_shape)
        start += bias1_shape[0]
        weight2 = params[start:start+weight2_shape[0]*weight2_shape[1]].view(weight2_shape).t()
        start += weight2_shape[0]*weight2_shape[1]
        bias2 = params[start:start+bias2_shape[-1]].view(bias2_shape)
        start += bias2_shape[0]
        assert start == len(params)
        
        out = self.graph_pred_linear[1](torch.matmul(h, weight1)+bias1)
        out = torch.matmul(out, weight2)+bias2
        return out
    
    def forward_(self, params, h, edge_index, batch=None):
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch
        
        assignment_all = self.graph_assignment_(params[0], h)
        
        graph_embeddings = []
        if self.assignment:
            assignments = []
        positives = []
        #negatives = []
        positive_penalties = []
        start = 0
        for b in range(batch.max()+1):
            #edge_start = []
            #edge_end = []
            node_index = (batch == b).nonzero().squeeze()
            
            assignment = assignment_all[node_index]
            group_features = torch.mm(torch.t(assignment), h[node_index])
            
            if self.assignment:
                assignments.append(assignment[0].unsqueeze(dim = 1))
            
            positive = torch.clamp(group_features[0].unsqueeze(dim = 0),-100,100)
            #negative = torch.clamp(group_features[1].unsqueeze(dim = 0),-100,100)
            
            graph_embedding = torch.mm(torch.t(assignment), h[node_index])
            graph_embedding = torch.mean(graph_embedding, dim = 0,keepdim= True)
            
            graph_embeddings.append(graph_embedding.view(1, self.num_emb))
            positives.append(positive.view(1, self.num_emb))
            #negatives.append(negative.view(1, self.num_emb))
            
            if self.connect:
                if h.is_cuda:
                    EYE = torch.ones(2).cuda()
                else:
                    EYE = torch.ones(2)
                for j in range(start, edge_index.shape[1]):
                    if edge_index[0][j] not in node_index or edge_index[1][j] not in node_index:
                        sub_edge_index_o = edge_index[:,start:j]
                        start = j
                        break
                    elif j == edge_index.shape[1]-1:
                        sub_edge_index_o = edge_index[:,start:]
                #sub_edge_index = torch.cat([torch.tensor(edge_start).unsqueeze(dim=0),torch.tensor(edge_end).unsqueeze(dim=0)],dim = 0)
                #sub_edge_index_o = sub_edge_index
                if b == batch.max() and j < edge_index.shape[1]-1:
                    print(node_index)
                    print(edge_index[:,j:])
                    print(edge_index)
                    raise ValueError('remain batch edge index in gib!')
                sub_edge_index = sub_edge_index_o - sub_edge_index_o.min()
                #sub_edge_index = sub_edge_index.cuda() if h.is_cuda else sub_edge_index
                Adj = to_dense_adj(sub_edge_index, max_num_nodes=len(node_index))[0]
                Adj.requires_grad = False
                new_adj = torch.mm(torch.t(assignment),Adj)
                new_adj = torch.mm(new_adj,assignment)
                
                normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
                norm_diag = torch.diag(normalize_new_adj)
                pos_penalty = self.mseloss(norm_diag, EYE)
                
                positive_penalties.append(pos_penalty.view(1))
                
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        positives = torch.cat(positives, dim=0)
        if self.pred:
            pred = self.graph_pred_linear_(params[1], torch.cat([graph_embeddings, positives], dim=0))
            if self.assignment:
                return pred, h, graph_embeddings, positives, assignments, positive_penalties
            else:
                return pred, h, graph_embeddings, positives, positive_penalties
        if self.assignment:
            return h, graph_embeddings, positives, assignments, positive_penalties
        else:
            return h, graph_embeddings, positives, positive_penalties

class SubGraph(nn.Module):
    def __init__(self, num_emb, num_layer, drop_ratio = 0.5, readout = 'add', con_weight = 5, connect = True, assignment = False, pred = True):
        super(SubGraph, self).__init__()
        self.gnn = MySchNet(hidden_channels=num_emb, num_interactions=num_layer, \
                          drop_ratio = drop_ratio, readout = readout, pred = False)
        self.gib = GIB(num_emb = num_emb, drop_ratio = drop_ratio, con_weight = con_weight, connect = connect, assignment=assignment, pred = pred)
        
    def forward(self, h, edge_index, edge_attr, batch=None):
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch
        
        _, node_features_2 = self.gnn(h, edge_index, edge_attr, batch)
        
        return self.gib(node_features_2, edge_index, batch)
    
    def forward_(self, params, h, edge_index, edge_attr, batch=None):
        batch = torch.zeros_like(h[:,0].long()) if batch is None else batch
        
        _, node_features_2 = self.gnn(h, edge_index, edge_attr, batch)
        
        return self.gib.forward_(params, node_features_2, edge_index, batch)

class Subgraph_Disc(torch.nn.Module):
    def __init__(self, emb_dim, threshold = 5):
        super(Subgraph_Disc, self).__init__()

        self.emb_dim = emb_dim
        self.input_size = 2 * self.emb_dim
        self.fc1 = torch.nn.Linear(self.input_size, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()
        
        self.params = None
        
        self.last_miloss = 0
        self.dcount = 0
        self.gcount = 0
        self.threshold = threshold

        torch.nn.init.constant(self.fc1.weight, 0.01)
        torch.nn.init.constant(self.fc2.weight, 0.01)
    
    def tune_disc(self, this_loss, flag):
        if this_loss>=self.last_miloss:
            self.dcount+=1
            self.gcount=0
        else:
            self.gcount+=1
            self.dcount=0
        self.last_miloss = this_loss
        if self.dcount >= self.threshold:
            return True
        elif self.gcount >= self.threshold:
            return False
        else:
            return flag
    
    def reset_tunecounter(self):
        self.dcount = 0
        self.gcount = 0
        self.last_miloss = 0
    
    def save_params(self):
        self.params = parameters_to_vector(self.parameters())
    
    def reset_params(self):
        vector_to_parameters(self.params, self.parameters())
    
    def forward(self, embeddings, positive):
        #print(embeddings)
        #print(positive)
        #print(self.fc1.weight)
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)
        pre = self.relu(self.fc1(cat_embeddings))
        pre = self.fc2(pre)

        return pre

class GraphDecoder(nn.Module):
    def forward(self, z, edge_index, batch = None, mean = True):
        batch = torch.zeros_like(z[:,0].long()) if batch == None else batch
        #batch = batch.cuda() if z.is_cuda else batch
        start = 0
        recon_loss = 0
        for i in range(batch.max()+1):
            #print(str(i)+':'+str(batch.max()))
            node_index = (batch == i).nonzero().squeeze()
            sub_z = z[node_index]
            #edge_start = []
            #edge_end = []
            #print('in loop')
            for j in range(start, edge_index.shape[1]):
                #print(str(j)+':'+str(edge_index.shape[1]))
                if edge_index[0][j] not in node_index or edge_index[1][j] not in node_index:
                    #print(edge_index[0][j])
                    #print(edge_index[1][j])
                    #print(node_index)
                    sub_edge_index_o = edge_index[:,start:j]
                    start = j
                    break
                elif j == edge_index.shape[1]-1:
                    sub_edge_index_o = edge_index[:,start:]
            #print('out loop')
            #sub_edge_index = torch.cat([torch.tensor(edge_start).unsqueeze(dim=0),torch.tensor(edge_end).unsqueeze(dim=0)],dim = 0)
            #sub_edge_index_o = sub_edge_index
            if i == batch.max() and j < edge_index.shape[1]-1:
                print(node_index)
                print(edge_index[:,j:])
                print(edge_index)
                raise ValueError('remain batch edge index in graph decoder!')
            sub_edge_index = sub_edge_index_o - sub_edge_index_o.min()
            prob = (sub_z[sub_edge_index[0]] * sub_z[sub_edge_index[1]]).sum(dim=1)
            adj = torch.sigmoid(prob)
            pos_loss = -torch.log(adj + EPS).mean()
            if torch.isnan(pos_loss).any():
                print(node_index)
                print(sub_edge_index_o)
                print(sub_edge_index)
                #print(sub_edge_index_)
                print(sub_z)
                print(adj)
                print(adj)
                print(pos_loss)
                raise ValueError('pos loss of recon loss contains nan!')
            
            sub_edge_index, _ = remove_self_loops(sub_edge_index)
            sub_edge_index, _ = add_self_loops(sub_edge_index)
            if (to_dense_adj(sub_edge_index)==0).any():
                sub_edge_index_ = negative_sampling(sub_edge_index, sub_z.size(0))
            elif sub_edge_index.is_cuda:
                sub_edge_index_ = torch.tensor([],dtype=torch.int64).cuda()
            else:
                sub_edge_index_ = torch.tensor([],dtype=torch.int64)
                
            if sub_edge_index_.shape[0] == 0:
                neg_loss = 0
            else:
                prob = (sub_z[sub_edge_index_[0]] * sub_z[sub_edge_index_[1]]).sum(dim=1)
                adj = torch.sigmoid(prob)
                neg_loss = -torch.log(1-adj +EPS).mean()
                if torch.isnan(neg_loss).any():
                    print(node_index)
                    print(sub_edge_index_o)
                    print(sub_edge_index)
                    print(sub_edge_index_)
                    print(sub_z)
                    print(adj)
                    print(neg_loss)
                    raise ValueError('neg loss of recon loss contains nan!')
            recon_loss+=(pos_loss+neg_loss)
        
        if mean:
            return recon_loss/((batch.max()+1).clone().detach_().item())
        else:
            return recon_loss

class GraphDecoder_(nn.Module):
    def __init__(self):
        super(GraphDecoder_, self).__init__()
        self.self_criterion = nn.BCEWithLogitsLoss()
    def build_negative_edges(self, x, edge_index):
        font_list = edge_index[0, ::2].tolist()
        back_list = edge_index[1, ::2].tolist()
        
        all_edge = {}
        for count, front_e in enumerate(font_list):
            if front_e not in all_edge:
                all_edge[front_e] = [back_list[count]]
            else:
                all_edge[front_e].append(back_list[count])
        
        negative_edges = []
        for num in range(x.size()[0]):
            if num in all_edge:
                for num_back in range(num, x.size()[0]):
                    if num_back not in all_edge[num] and num != num_back:
                        negative_edges.append((num, num_back))
            else:
                for num_back in range(num, x.size()[0]):
                    if num != num_back:
                        negative_edges.append((num, num_back))

        negative_edge_index = torch.tensor(np.array(random.sample(negative_edges, len(font_list))).T, dtype=torch.long)

        return negative_edge_index
    
    def forward(self, x, edge_index):
        positive_score = torch.sum(x[edge_index[0, ::2]] * x[edge_index[1, ::2]], dim = 1)
        negative_edge_index = self.build_negative_edges(x, edge_index)
        negative_score = torch.sum(x[negative_edge_index[0]] * x[negative_edge_index[1]], dim = 1)
        
        self_loss = torch.sum(self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(negative_score, torch.zeros_like(negative_score)))/negative_edge_index[0].size()[0]
        
        return self_loss

class ParamDiscriminator(nn.Module):
    def __init__(self, param_num, diff = 'wf'):
        super(ParamDiscriminator, self).__init__()
        #self.num_emb = num_emb
        self.param_num = param_num
        self.encoder = nn.Sequential(
                    nn.Linear(param_num,1024),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(1024,512),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(512,256),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(256,128),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(128,1),
                )
        if diff not in ['cf', 'wf']:
            raise ValueError('not implemented gradient penalty.')
        else:
            self.diff = diff
        torch.nn.init.kaiming_normal_(self.encoder[0].weight)
        torch.nn.init.kaiming_normal_(self.encoder[2].weight)
        torch.nn.init.kaiming_normal_(self.encoder[4].weight)
        torch.nn.init.kaiming_normal_(self.encoder[6].weight)
        torch.nn.init.kaiming_normal_(self.encoder[8].weight)
    
    def gradient_penalty(self, xr, xf, yr, yf):
        # [b, 1] => [b, 2]
        batchsz = xr.shape[0]
        t = torch.rand(batchsz, 1).cuda()
        t = t.expand_as(xr)
        
        #在真实数据和生成的做插值
        mid = t * xr + ((1 - t) * xf)
        #做导数
        mid.requires_grad_()
        pred = self.encoder(mid)
        if self.diff == 'wf':
            grads = torch.autograd.grad(outputs=pred, inputs=mid,
                                  grad_outputs=torch.ones_like(pred),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            gp = torch.pow((grads.norm(2, dim=1) - 1) , 2).mean()
        elif self.diff == 'cf':
            gp = torch.pow(((torch.abs(yr-yf)/((xr-xf).norm(2, dim = 1))) - 1),2).mean()
        else:
            raise ValueError('unmatched diff type.')
        #2范数越接近于1越好
        
        return gp
    
    def forward(self, xr, xf):
        yr = self.encoder(xr)
        yf = self.encoder(xf)
        gp = self.gradient_penalty(xr, xf, yr, yf)
        
        return yr, yf, gp

class ParameterEncoder(nn.Module):
    def __init__(self, num_emb, param_num):
        super(ParameterEncoder, self).__init__()
        self.num_emb = num_emb
        self.param_num = param_num
        self.encoder = nn.Sequential(
                    nn.Linear(param_num,1024),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(1024,512),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(512,256),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(256,128),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(128,num_emb),
                    nn.LeakyReLU(inplace = True),
                )
        torch.nn.init.kaiming_normal_(self.encoder[0].weight)
        torch.nn.init.kaiming_normal_(self.encoder[2].weight)
        torch.nn.init.kaiming_normal_(self.encoder[4].weight)
        torch.nn.init.kaiming_normal_(self.encoder[6].weight)
        torch.nn.init.kaiming_normal_(self.encoder[8].weight)
    def forward(self, params):
        h=self.encoder(params)
        return h
        
class ParameterDecoder(nn.Module):
    def __init__(self, num_emb, param_num, fixedz = True):
        super(ParameterDecoder, self).__init__()
        self.num_emb = num_emb
        self.param_num = param_num
        self.encoder = nn.Linear(num_emb, 2*num_emb)
        self.generator = nn.Sequential(
                    nn.Linear(num_emb,128),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(128,256),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(256,512),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(512,1024),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(1024,param_num),
                )
        torch.nn.init.kaiming_normal_(self.encoder.weight)
        torch.nn.init.kaiming_normal_(self.generator[0].weight)
        torch.nn.init.kaiming_normal_(self.generator[2].weight)
        torch.nn.init.kaiming_normal_(self.generator[4].weight)
        torch.nn.init.kaiming_normal_(self.generator[6].weight)
        torch.nn.init.kaiming_normal_(self.generator[8].weight)
    def forward(self, bproteins):
        h_=self.encoder(bproteins)
        mu, log_var = h_.chunk(2, dim=-1)
        sigma = torch.exp(log_var * 0.5)
        h = (mu + sigma * torch.randn_like(sigma).cuda())
        params = self.generator(h)
        return params, mu, log_var

class ParameterDecoder_(nn.Module):
    def __init__(self, num_emb, param_num):
        super(ParameterDecoder_, self).__init__()
        self.num_emb = num_emb
        self.param_num = param_num
        self.generator = nn.Sequential(
                    nn.Linear(num_emb,128),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(128,256),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(256,512),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(512,1024),
                    nn.LeakyReLU(inplace = True),
                    nn.Linear(1024,param_num),
                )
        torch.nn.init.kaiming_normal_(self.generator[0].weight)
        torch.nn.init.kaiming_normal_(self.generator[2].weight)
        torch.nn.init.kaiming_normal_(self.generator[4].weight)
        torch.nn.init.kaiming_normal_(self.generator[6].weight)
        torch.nn.init.kaiming_normal_(self.generator[8].weight)
    def forward(self, bproteins):
        params = self.generator(bproteins)
        return params

class ParamDecoder(nn.Module):
    def __init__(self, num_emb, param_shape):
        super(ParamDecoder, self).__init__()
        self.num_emb = num_emb
        self.param_shape = param_shape
        #self.encoder = ParameterEncoder(num_emb, param_shape)
        self.decoder = ParameterDecoder(num_emb, param_shape)
        self.loss_func = nn.MSELoss()
    
    def forward(self, pvec, params):
        #batchsz = pvec.shape[0]
        params_, mu_, log_var_ = self.decoder(pvec)
        kld_ = 0.5 * torch.sum(torch.exp(log_var_) + torch.pow(mu_, 2) - 1. - log_var_)
        print('\tPvec:'+str(pvec))
        print('\tParams mean:'+str(params.mean(dim = -1)))
        print('\tParams_ mean:'+str(params_.mean(dim = -1)))
        print('\tParams std:'+str(str(params.std(dim = -1))))
        print('\tParams_ std:'+str(str(params_.std(dim = -1))))
        loss = self.loss_func(params_, params) + kld_
        return loss, params_

class ParamDecoder_(nn.Module):
    def __init__(self, num_emb, param_shape):
        super(ParamDecoder_, self).__init__()
        self.num_emb = num_emb
        self.param_shape = param_shape
        self.decoder = ParameterDecoder_(num_emb, param_shape)
        self.loss_func = nn.MSELoss()
    
    def forward(self, pvec, params):
        params_= self.decoder(pvec)
        print('\tPvec:'+str(pvec))
        print('\tParams mean:'+str(params.mean(dim = -1)))
        print('\tParams_ mean:'+str(params_.mean(dim = -1)))
        print('\tParams std:'+str(str(params.std(dim = -1))))
        print('\tParams_ std:'+str(str(params_.std(dim = -1))))
        loss = self.loss_func(params_, params)
        return loss, params_

class ParamVAE(nn.Module):
    def __init__(self, num_emb, param_shape):
        super(ParamVAE, self).__init__()
        self.num_emb = num_emb
        self.param_shape = param_shape
        self.encoder = ParameterEncoder(num_emb, param_shape)
        self.decoder = ParameterDecoder(num_emb, param_shape)
        self.loss_func = nn.MSELoss()
    
    def forward(self, pvec, params):
        #batchsz = pvec.shape[0]
        pvec_ = self.encoder(params)
        params_, mu_, log_var_ = self.decoder(pvec)
        '''
        kld_ = 0.5 * torch.mean(
            torch.pow(mu_, 2) +
            torch.pow(sigma_, 2) -
            torch.log(1e-8 + torch.pow(sigma_, 2)) - 1
        )
        '''
        #kld_ = - 0.5 * torch.mean(1 + sigma_ - mu_.pow(2) - sigma_.exp())
        kld_ = 0.5 * torch.sum(torch.exp(log_var_) + torch.pow(mu_, 2) - 1. - log_var_)
        pvec__ = self.encoder(params_.clone())
        params__, mu__, log_var__ = self.decoder(pvec_.clone())
        '''
        kld__ = 0.5 * torch.mean(
            torch.pow(mu__, 2) +
            torch.pow(sigma__, 2) -
            torch.log(1e-8 + torch.pow(sigma__, 2)) - 1
        ) / (batchsz*28*28)
        '''
        #kld__ = - 0.5 * torch.mean(1 + sigma__ - mu__.pow(2) - sigma__.exp())
        kld__ = 0.5 * torch.sum(torch.exp(log_var__) + torch.pow(mu__, 2) - 1. - log_var__)
        print('\tPvec:'+str(pvec))
        print('\tParams mean:'+str(params.mean(dim = -1)))
        print('\tParams_ mean:'+str(params_.mean(dim = -1)))
        print('\tParams std:'+str(str(params.std(dim = -1))))
        print('\tParams_ std:'+str(str(params_.std(dim = -1))))
        loss = self.loss_func(pvec_, pvec) + self.loss_func(pvec__, pvec) + kld_ + kld__ + self.loss_func(params_, params) + self.loss_func(params__, params)
        return loss, (params_, params__)

class ParamVAE_(nn.Module):
    def __init__(self, num_emb, param_shape):
        super(ParamVAE_, self).__init__()
        self.num_emb = num_emb
        self.param_shape = param_shape
        self.encoder = ParameterEncoder(num_emb, param_shape)
        self.decoder = ParameterDecoder_(num_emb, param_shape)
        self.loss_func = nn.MSELoss()
    
    def forward(self, pvec, params):
        #batchsz = pvec.shape[0]
        pvec_ = self.encoder(params)
        params_ = self.decoder(pvec)
        pvec__ = self.encoder(params_)
        params__ = self.decoder(pvec_)
        loss = self.loss_func(params_, params) + self.loss_func(pvec_, pvec) + \
            self.loss_func(params__, params) + self.loss_func(pvec__,pvec)
        print('\tPvec:'+str(pvec))
        print('\tParams mean:'+str(params.mean(dim = -1)))
        print('\tParams_ mean:'+str(params_.mean(dim = -1)))
        print('\tParams std:'+str(str(params.std(dim = -1))))
        print('\tParams_ std:'+str(str(params_.std(dim = -1))))
        return loss, (params_, params__)

class SVM(nn.Module):
    def __init__(self, emb_dim):
        super(SVM, self).__init__()
        self.emb_dim = emb_dim
        self.lin = nn.Linear(self.emb_dim, 1)
        
    def loss_func(self, output, y):
        y = torch.where(y==0, torch.ones_like(y)*-1, y) if 0 in y else y
        loss = torch.mean(torch.clamp(1 - output.t() * y, min=0))
        return loss
        
    def forward(self, x, y):
        out = self.lin(x)
        loss = self.loss_func(out, y)
        return loss
    
    def sign(self, x):
        #x = self.lin(x)
        pred = torch.where(x>=0, torch.ones_like(x), x)
        pred = torch.where(pred<0, torch.zeros_like(pred), pred)
        return x, pred
    
    def test(self, x):
        return self.sign(self.lin(x))


class LR(nn.Module):
    def __init__(self, emb_dim):
        super(LR, self).__init__()
        self.emb_dim = emb_dim
        self.lin = nn.Sequential(
            nn.Linear(self.emb_dim, 1),
            nn.Sigmoid()
        )
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        out = self.lin(x)
        return self.loss_func(torch.cat([1-out, out],dim = -1), y)
    
    def test(self, x):
        x = self.lin(x)
        pred = torch.where(x>=0.5, torch.ones_like(x), x)
        pred = torch.where(pred<0.5, torch.zeros_like(pred), pred)
        return x, pred
    
class GraphDTA(nn.Module):
    def __init__(self, emb_dim):
        super(GraphDTA, self).__init__()
        self.emb_dim = emb_dim
        self.gnn = MySchNet(hidden_channels=emb_dim, num_interactions=6, \
                          drop_ratio = 0.5, readout = 'add', pred = False)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1024, 512, 10, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 10, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 10, 1),
            nn.LeakyReLU(),
            nn.MaxPool1d(973),
        )
        
        self.lin = nn.Sequential(
            nn.Linear(256, 32) ,
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def run(self, h, edge_index, edge_attr, batch, emb_r):
        emb_l, _ = self.gnn(h, edge_index, edge_attr, batch)
        
        #emb_r = torch.zeros(batch.max()+1, emb_r.shape[0], emb_r.shape[1]).copy_(emb_r.unsqueeze(dim = 0))
        #emb_r = emb_r.cuda() if 
        emb_r = torch.cat([emb_r.unsqueeze(dim = 0)]*(batch.max()+1), dim = 0)
        emb_r = self.cnn(emb_r.transpose(1,2)).squeeze(dim = -1)
        
        emb = torch.cat([emb_l.view(batch.max()+1,128), emb_r.view(batch.max()+1,128)], dim = -1)
        
        out = self.lin(emb)
        
        return out
    
    def forward(self, h, edge_index, edge_attr, batch, emb_r, y):
        out = self.run(h, edge_index, edge_attr, batch, emb_r)
        loss = self.loss_func(out, y)
        return loss
    
    def test(self, h, edge_index, edge_attr, batch, emb_r):
        out = torch.softmax(self.run(h, edge_index, edge_attr, batch, emb_r), dim = -1)
        pred = out.argmax(dim = -1)
        return out[:,1], pred
        

class DeepConvDTI(nn.Module):
    def __init__(self, emb_dim):
        super(DeepConvDTI, self).__init__()
        self.emb_dim = emb_dim
        self.gnn = nn.Linear(256, 128)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1024, 512, 10, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 10, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 10, 1),
            nn.LeakyReLU(),
            nn.MaxPool1d(973),
        )
        
        self.lin = nn.Sequential(
            nn.Linear(256, 32) ,
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def run(self, fps, emb_r):
        batch_size=fps.shape[0]
        emb_l = self.gnn(fps)
        
        #emb_r = torch.zeros(batch.max()+1, emb_r.shape[0], emb_r.shape[1]).copy_(emb_r.unsqueeze(dim = 0))
        #emb_r = emb_r.cuda() if 
        #emb_r = torch.cat([emb_r.unsqueeze(dim = 0)]*(batch_size), dim = 0)
        emb_r = self.cnn(emb_r.transpose(1,2)).squeeze(dim = -1)
        
        emb = torch.cat([emb_l.view(batch_size,128), emb_r.view(batch_size,128)], dim = -1)
        
        out = self.lin(emb)
        
        return out
    
    def forward(self, fps, emb_r, y):
        out = self.run(fps, emb_r)
        loss = self.loss_func(out, y)
        return loss
    
    def test(self, fps, emb_r):
        out = torch.softmax(self.run(fps, emb_r), dim = -1)
        pred = out.argmax(dim = -1)
        return out[:,1], pred
        



