import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing
import numpy as np
import os
import time
import random
from tqdm import tqdm
from scipy.io import loadmat
from typing import List
from sklearn.metrics import confusion_matrix
from GraphVAE import VAE_LL_loss
from utilis import Calculate_TC

criterion = nn.CrossEntropyLoss()

def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None

parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
parser.add_argument('--epochs', type=int, default=310,
                        help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=50,
                        help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument("--emb_normlize", type = bool, default=  False, help="mlp hidden dims")
parser.add_argument("--data_split_ratio", type=float, default= [0.8, 0.1, 0.1], help="data seperation")
parser.add_argument("--latent_dim", type=int, default=  [128,128,128], help="classifier hidden dims")
parser.add_argument('--readout', type=str, default="mean", choices=["sum", "average", "max"],
                        help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
parser.add_argument("--mlp_hidden", type=int, default=  [128,128], help="mlp hidden dims")
parser.add_argument("--GVAE_hidden_dim", type = int, default= 120, help="mlp hidden dims")
parser.add_argument("--mi_weight", type=float, default= 0.001, help="classifier hidden dims")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Adam weight decay. Default is 5*10^-5.")
parser.add_argument("--input_dim", type=int, default=116)
parser.add_argument("--output_dim", type=int, default=2)
parser.add_argument("--lambda1", type=float, default=0.001)
parser.add_argument("--lambda2", type=float, default=0.001)
parser.add_argument("--lambda3", type=float, default=0.001)
parser.add_argument("--num_prototypes", type=int, default=2)
args = parser.parse_args()

#set up seeds and gpu device
torch.manual_seed(0)
np.random.seed(0)    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# --- load data ---
from utilis import get_dataloader,load_dataset
graph_filename = "data/ABIDE.mat"
graph = loadmat(graph_filename)
dataset = load_dataset(graph)
dataloader = get_dataloader(dataset, args.batch_size, data_split_ratio=args.data_split_ratio, seed=args.seed)

# --- train/load GCE ---
from GraphVAE import GraphEncoder,GraphDecoder
encoder = GraphEncoder(args.input_dim, args.GVAE_hidden_dim, device).to(device)
decoder = GraphDecoder(args.GVAE_hidden_dim, args.input_dim).to(device)

# --- train/load subgraph ---
from pre_subgraph import Prot_subgraph
from GIN_classifier import GINNet
model = GINNet(args.input_dim, args.output_dim, args, device).to(device)
SG_model = Prot_subgraph(device, encoder, decoder, model, args.GVAE_hidden_dim, args.num_prototypes, args.dropout).to(device)


def train(args, model, device, train_graphs, eval_graphs,test_graphs, SG_model,opt,optimizer, k, num_prototype):

    model.eval()
    SG_model.eval()
    encoder.train()
    decoder.train()
    if k<20:
        for graph in train_graphs: 
            x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
            z, mu, logvar= encoder(graph)
            Xhat, adj = decoder(z)
            # #print(Xhat.shape)
            nll = VAE_LL_loss(graph.x, Xhat, logvar, mu, device)
            graph_prot = torch.zeros(args.num_prototypes, len(z[:,0]),round(args.GVAE_hidden_dim /args.num_prototypes)).to(device)
            for k in range(args.num_prototypes):
                prot_size_up = round(args.GVAE_hidden_dim * k/args.num_prototypes)
                prot_size_de = round(args.GVAE_hidden_dim * (k+1)/args.num_prototypes)
                edge = z[:,prot_size_up:prot_size_de]
                print(edge.shape)
                graph_prot[k,:,:]= edge
            TC_loss = Calculate_TC(graph_prot, args.num_prototypes)
            prototype_loss = nll + 0.001 * TC_loss
            print(prototype_loss)
            if opt is not None:
                opt.zero_grad()
                prototype_loss.backward()    
                opt.step()
    else:
        model.train()
        SG_model.train()
        encoder.train()
        decoder.train()
        acc_accum = 0
        num = 0
        for graph in train_graphs: 
            x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
            logits, loss = SG_model(graph,args.lambda2)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()      
                optimizer.step()
            

            loss = loss.detach().cpu().numpy()
            pred = logits.max(1, keepdim=True)[1]
            labels = torch.LongTensor(graph.y).to(device)
            print(pred.shape)
            print(labels.shape)
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(graph.y))
            acc_accum = acc_accum + acc
            num = num + 1
                    
        acc_train = acc_accum/num
        print("classification loss: %f" %(loss))
        print("accuracy train: %f" %(acc_train))
        
        acc_eval = evaluate(args, eval_graphs, model, SG_model, device)
        print("accuracy test: %f" %(acc_eval))

        acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test = test(args, test_graphs, model, SG_model, device)
        print("accuracy test: %f" %(acc_test))

        filename="ABIDE3.txt"
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("%f %f %f %f %f %f %f %f %f" % (loss, acc_train, acc_eval, acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test))
                f.write("\n")
        else:
            with open(filename, 'a+') as f:
                f.write("%f %f %f %f %f %f %f %f %f" % (loss, acc_train, acc_eval, acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test))
                f.write("\n")

        return loss

def evaluate(args, data, model, SG_model, device): 

    acc_accum = 0
    num = 0
    for graph in data: 
        x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
        logits, loss = SG_model(graph, args.lambda2)
        pred = logits.max(1, keepdim=True)[1]
        labels = torch.LongTensor(graph.y).to(device)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        acc = correct / float(len(graph.y))
        acc_accum = acc_accum + acc
        num = num + 1
    acc_test = acc_accum/num

    return acc_test

def calc_performance_statistics(y_pred, y):

    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    acc = (TN + TP) / N
    sen = TP / (TP + FN)
    spc = TN / (TN + FP)
    prc = TP / (TP + FP)
    f1s = 2 * (prc * sen) / (prc + sen)
    mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))

    return acc, sen, spc, prc, f1s, mcc

def test(args, data, model, SG_model, device): 

    model.eval()
    SG_model.eval()

    acc_accum = 0
    sen_accum = 0
    spc_accum = 0
    prc_accum = 0
    f1s_accum = 0
    mcc_accum = 0
    num = 0

    for graph in data: 
        x, edge_index, batch = graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device)
        logits, loss = SG_model(graph,args.lambda2)
        #compute gradient
        pred = logits.max(1, keepdim=True)[1]
        labels = torch.LongTensor(graph.y).to(device)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        if len(labels)>1:
            test_acc, test_sen, test_spc, test_prc, test_f1s, test_mcc = calc_performance_statistics(pred,labels)
            acc = correct / float(len(graph.y))
            acc_accum = acc_accum + acc
            sen_accum = sen_accum + test_sen
            spc_accum = spc_accum + test_spc
            prc_accum = prc_accum + test_prc
            f1s_accum = f1s_accum + test_f1s
            mcc_accum = mcc_accum + test_mcc
            num = num + 1
        acc_test = acc_accum/num
        sen_test = sen_accum/num
        spc_test = spc_accum/num
        prc_test = prc_accum/num
        f1s_test = f1s_accum/num
        mcc_test = mcc_accum/num

    return acc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test

for  fold_idx in range(0,3):

    encoder = GraphEncoder(args.input_dim, args.GVAE_hidden_dim, device).to(device)
    decoder = GraphDecoder(args.GVAE_hidden_dim, args.input_dim).to(device)
    model = GINNet(args.input_dim, args.output_dim, args, device).to(device)
    SG_model = Prot_subgraph(device, encoder, decoder, model, args.GVAE_hidden_dim, args.num_prototypes, args.dropout).to(device)

    dataloader = get_dataloader(dataset, args.batch_size, data_split_ratio=args.data_split_ratio, seed=random.randint(0,1000))

    opt_params = list(decoder.parameters()) + list(encoder.parameters())
    opt = torch.optim.Adam(opt_params, lr=args.lr , weight_decay=5e-4)
    params = list(model.parameters()) + list(SG_model.parameters())
    optimizer = torch.optim.Adam( params, lr =args.lr , weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        

        avg_loss = train(args, model, device,  dataloader['train'],dataloader['eval'],dataloader['test'], SG_model,opt,optimizer,epoch, args.num_prototypes)

        scheduler.step()
