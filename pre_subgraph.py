from pickle import FALSE, TRUE
import torch
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import numpy as np
from torch_geometric.nn import MessagePassing
from GraphVAE import VAE_LL_loss
from utilis import Calculate_TC, reyi_entropy

criterion = nn.CrossEntropyLoss()
EPS = 1e-15

# Multi-layer Perceptron with Bernoulli mask
# noinspection PyArgumentList
class Prot_subgraph(nn.Module):
    def __init__(self,device, encoder, decoder, classifier, GVAE_hidden_dim, num_prototypes, final_dropout):
        super(Prot_subgraph, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim =116
        self.classifier = classifier
        self.device = device
        self.GVAE_hidden_dim = GVAE_hidden_dim
        self.num_prototypes = num_prototypes
        self.output_dim = 2
        self.prototype_shape = (self.num_prototypes, 128)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        self.last_layer = nn.Linear(self.num_prototypes, self.output_dim)
        self.final_dropout = final_dropout

    def get_retain_mask(self, drop_probs, shape, tau):
        tau = 1.0
        uni = torch.rand(shape).to(self.device)
        eps = torch.tensor(1e-8).to(self.device)
        tem = (torch.log(drop_probs + eps) - torch.log(1 - drop_probs + eps) + torch.log(uni + eps) - torch.log(
            1.0 - uni + eps))
        mask = 1.0 - torch.sigmoid(tem / tau)
        return mask

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs= gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        return graph

    def clear_masks(self,model):
        """ clear the edge weights to None """
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None


    def set_masks(self,model, edgemask):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = edgemask

    def prototype_subgraph_similarity(self, x, prototype):
        distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
        similarity = torch.log((distance + 1) / (distance + 1e-4))
        return similarity,distance

    def prot_decoder(self, z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)

    def forward(self, graph, lambda2):
        loss = 0
        sparse_loss = 0
        entropy_loss = 0
        sim_loss = 0
        prototype_edge = []
        edge_index, batch,x = graph.edge_index.to(self.device), graph.batch.to(self.device), graph.x.to(self.device)
        # logits, probs, node_emb, graph_emb, _ = self.classifier(x,edge_index,batch)
        z, mu, logvar= self.encoder(graph)
        #Xhat, adj = self.decoder(z)
        # #print(Xhat.shape)
        #nll = VAE_LL_loss(graph.x, Xhat, logvar, mu, self.device)

        # nodesize = graph.x.shape[0]
        # edgesize = len(edge_index[0])
        graph_prot = torch.zeros(self.num_prototypes, len(z[:,0]),round(self.GVAE_hidden_dim /self.num_prototypes)).to(self.device)
        similarity=[]
        distance = []
        for k in range(self.num_prototypes):

            prot_size_up = round(self.GVAE_hidden_dim * k/self.num_prototypes)
            prot_size_de = round(self.GVAE_hidden_dim * (k+1)/self.num_prototypes)
            edge = z[:,prot_size_up:prot_size_de]
            graph_prot[k,:,:]= edge
            edge_mask = self.prot_decoder(edge)
            aedge = edge_mask[edge_index[0], edge_index[1]].to(self.device)
            #aedge =  self._sample_graph(aedge)
            aedge_mask = aedge.unsqueeze(1)
            prototype_edge.append(aedge_mask)
             
            aedge =  torch.sigmoid(F.gumbel_softmax(aedge,tau = 0.1))
            self.clear_masks(self.classifier)
            self.set_masks(self.classifier, aedge)
            _, prot_emb,_ = self.classifier(x,edge_index,batch)
            sim = torch.norm(prot_emb - self.prototype_vectors[k])
            sim_loss += sim
            similarity1, distance1 = self.prototype_subgraph_similarity(prot_emb,self.prototype_vectors[k])
            similarity.append(similarity1)
            distance.append(distance1)
            sparse_loss += 0.005 * aedge.sum()/10
            m = aedge
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            #ent = reyi_entropy(m)
            entropy_loss += ent.mean()

        prototype_activations = torch.cat(tuple(similarity),dim = 1)
        prototype_edge = torch.cat(tuple(prototype_edge),dim = 1)
        distance = torch.cat(tuple(distance),dim = 1)
        # print(prototype_activations.shape)
        logits = self.last_layer(prototype_activations)
        # TC_loss = Calculate_TC(graph_prot, self.num_prototypes)
        labels = torch.LongTensor(graph.y).to(self.device)
        #TC_loss = Calculate_TC(graph_prot, self.num_prototypes)
        #loss = criterion(logits, labels) + sparse_loss + lambda3 * entropy_loss +  lambda2 * sim_loss + 0.001 * nll
        loss = criterion(logits, labels) + 0.0001 * (sparse_loss + entropy_loss) +  lambda2 * sim_loss 
        # loss = criterion(logits, labels)
        # return prototype_edge , distance, prototype_activations
        return logits, loss
