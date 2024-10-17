from tkinter.tix import TCL_TIMER_EVENTS
import torch
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch_geometric.utils import dense_to_sparse
import math

eps = 1e-8
def load_dataset(graph):
    print('loading data')
    num_graphs=graph["label"].size
    label1=graph["label"]
    label=np.append(label1,label1)
    data_list = []
    for i in range(num_graphs):
        node_features = torch.FloatTensor(graph["graph_struct"][0][i][1])
        tepk = node_features.reshape(-1,1)
        tepk, indices = torch.sort(abs(tepk), dim=0, descending=True)
        mk = tepk[int(node_features.shape[0] * node_features.shape[0] * 0.2 - 1)]
        edge = torch.Tensor(np.where(node_features > mk, 1, 0))
        data_example = Data(x=node_features,edge_index=dense_to_sparse(edge)[0],y=label[i])
        data_list.append(data_example)

    return data_list


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=None):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size = num_eval, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size = num_test, shuffle=False)
    return dataloader

def pairwise_distances(x):
    #x should be two dimensional
    if x.dim()==1:
        x = x.unsqueeze(1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_sigma(Z_numpy):   

    if Z_numpy.dim()==1:
        Z_numpy = Z_numpy.unsqueeze(1)
    Z_numpy = Z_numpy.cpu().detach().numpy()
    #print(Z_numpy.shape)
    k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
    sigma = np.mean(np.mean(np.sort(k[:, :10], 1))) 
    if sigma < 0.1:
        sigma = 0.1
    return sigma 

def calculate_gram_mat(x):
    dist= pairwise_distances(x)
    sigma = calculate_sigma(x)**2
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)

def reyi_entropy(x):
    alpha = 1.8
    k = calculate_gram_mat(x)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(data,num_prot):
    alpha = 1.8
    k = 1
    for i in range(num_prot):
        k = k * calculate_gram_mat(data[i,:,:])
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def Calculate_TC(data,num_prot):
    alpha = 2
    entropy = 0.0
    for i in range(num_prot):
        entropy = entropy + reyi_entropy(data[i,:,:])
    joint_en = joint_entropy(data,num_prot)
    TC = entropy - joint_en

    return TC


def separate_data(dataset,data_split_ratio=None, seed=None):
    
    num_train = int(data_split_ratio[0] * len(dataset))
    num_eval = int(data_split_ratio[1] * len(dataset))
    num_test = len(dataset) - num_train - num_eval

    train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                        generator=torch.Generator().manual_seed(seed))

    return train, eval, test

def separate_site_ASD(dataset,site,site_idx, seed=None):
    
    site = torch.LongTensor(site)
    test_idx = torch.nonzero(site == site_idx)[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_idx)[:,0].numpy().tolist()

    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)

    return train, test

def separate_site_MDD(dataset,site, site_idx, seed=None):
    
    site_name = [1, 2, 4, 7, 8, 9, 10, 11, 13, 14, 15, 17, 3, 5, 6, 12, 16]
    site = torch.LongTensor(site)
    print(site_name[site_idx])
    test_idx = torch.nonzero(site == site_name[site_idx])[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_name[site_idx])[:,0].numpy().tolist()
    print(test_idx)
    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)

    return train, test

def get_dataloader_site(dataset, site, site_idx,batch_size):
    
    site = torch.LongTensor(site)
    test_idx = torch.nonzero(site == site_idx)[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_idx)[:,0].numpy().tolist()

    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)
    print(test_idx)

    dataloader = dict()
    test_batch_size = len(test_idx)
    print(test_batch_size)
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return dataloader

def get_dataloader_site_MDD(dataset, site, site_idx,batch_size):
    
    site_name = [1, 2, 4, 7, 8, 9, 10, 11, 13, 14, 15, 17, 3, 5, 6, 12, 16]
    site = torch.LongTensor(site)
    print(site_name[site_idx])
    test_idx = torch.nonzero(site == site_name[site_idx])[:,0].numpy().tolist()
    train_idx = torch.nonzero(site != site_name[site_idx])[:,0].numpy().tolist()
    print(test_idx)
    train = Subset(dataset, train_idx)
    test = Subset(dataset, test_idx)
    test_batch_size = len(test_idx)
    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return dataloader