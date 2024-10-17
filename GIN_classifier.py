import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch.autograd import Variable

def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


# GIN
class GINNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_args, device):
        super(GINNet, self).__init__()
        self.latent_dim = model_args.latent_dim
        self.mlp_hidden = model_args.mlp_hidden
        self.emb_normlize = model_args.emb_normlize
        self.device = device
        self.num_gnn_layers = len(self.latent_dim)
        self.num_mlp_layers = len(self.mlp_hidden) + 1
        self.dense_dim = self.latent_dim[-1]
        self.readout_layers = get_readout_layers(model_args.readout)
        self.inter_dim = 40
        self.dense_dim2 = self.inter_dim * self.inter_dim
        self.BiMap = nn.Linear(self.dense_dim, self.inter_dim, bias=False)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, self.latent_dim[0], bias=False),
            nn.BatchNorm1d(self.latent_dim[0]),
            nn.ReLU(),
            nn.Linear(self.latent_dim[0], self.latent_dim[0], bias=False),
            nn.BatchNorm1d(self.latent_dim[0])),
            train_eps=True))

        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(GINConv(nn.Sequential(
                nn.Linear(self.latent_dim[i-1], self.latent_dim[i], bias=False),
                nn.BatchNorm1d(self.latent_dim[i]),
                nn.ReLU(),
                nn.Linear(self.latent_dim[i], self.latent_dim[i], bias=False),
                nn.BatchNorm1d(self.latent_dim[i])),
                train_eps=True)
            )

        self.gnn_non_linear = nn.ReLU()

        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.dense_dim,
                                       model_args.mlp_hidden[0]))
            for i in range(1, self.num_mlp_layers-1):
                self.mlps.append(nn.Linear(self.mlp_hidden[i-1], self.mlp_hidden[1]))
            self.mlps.append(nn.Linear(self.mlp_hidden[-1], output_dim))
        else:
            self.mlps.append(nn.Linear(self.dense_dim,output_dim))
        self.dropout = nn.Dropout(model_args.dropout)
        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ELU()
        # self.last_layer = nn.Linear(self.num_prototypes, output_dim,
        #                             bias=False)  # do not use bias

    # def random_feature(self, x):
    #     r = torch.rand(size=(len(x), 1)).to(x.device)
    #     x = torch.cat([x, r], dim=-1)
    #     return x

    def forward(self, x, edge_index, batch,edge_weight = None):
        
        # x = self.random_feature(x)
        x,edge_index = x.to(self.device),edge_index.to(self.device)
        for i in range(self.num_gnn_layers):
            if edge_weight == None:
                x = self.gnn_layers[i](x, edge_index)
            else:
                x = self.gnn_layers[i](x, edge_index, edge_weight)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x
        

        # graph_size = int(len(x[:,0])/116)
        # node_emb = torch.split(x, 116, dim=0)
        # batch_graphs = torch.zeros(graph_size, self.dense_dim2).to(self.device)
        # batch_graphs = Variable(batch_graphs)


        # for g_i in range(graph_size):
        #     cur_node_embeddings = node_emb[g_i]
        #     cur_node_embeddings = self.BiMap(cur_node_embeddings)
        #     cur_graph_embeddings = torch.matmul(cur_node_embeddings.t(), cur_node_embeddings)
        #     batch_graphs[g_i] = cur_graph_embeddings.view(self.dense_dim2)
        
        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(x, batch))
        x = torch.cat(pooled, dim=-1)
        graph_emb = x

        for i in range(self.num_mlp_layers):
            x = self.mlps[i](x)
            x = self.mlp_non_linear(x)
            x = self.dropout(x)

        logits = x
        probs = self.Softmax(logits)

        return logits, graph_emb, node_emb
