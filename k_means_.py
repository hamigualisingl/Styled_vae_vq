import os
import numpy as np
import torch
from tqdm import tqdm
from torch import distributed
import torch.distributed as dist
def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        iteration_=500,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state)
       #
        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol and iteration>iteration_:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()

def pairwis(data1, data2, device=torch.device('cpu')):
    #data1, data2 = data1.to(device), data2.to(device)
    d = torch.sum(data1**2, dim=1, keepdim=True) + \
            torch.sum(data2**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', data1, data2.T)
    
    return d
def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

def kmeans_dist(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        iteration_=50,
        device=torch.device('cpu'),
        rank=0,
        batch_size=8192
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = torch.zeros(num_clusters,X.shape[-1], device=device)

    if dist.get_rank() == 0:
        initial_state = initialize(X, num_clusters).to(device)
        #print(initial_state)
    #dist.broadcast(initial_state, src=0)
    iteration = 0
    #tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dist.broadcast(initial_state, src=0)####万全之策
        choice_clusters = []
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            X_batch = X[start_idx:end_idx]
            dis = pairwis(X_batch, initial_state, device=device)
            choice_cluster = torch.argmin(dis, dim=1)
            choice_clusters.append(choice_cluster)
        choice_cluster = torch.cat(choice_clusters, dim=0)
        torch.distributed.barrier()
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()#.to(device)######获取有哪些样本选择了这个中心
            selected = torch.index_select(X, 0, selected)#####获取有哪些样本选择了这个中心
            new_center = selected.mean(dim=0)######取均值，现在是每张卡都有所有中心
            #([0.4492, 0.6931], device='cuda:0'),([0.4703, 0.7043], device='cuda:1')
            dist.all_reduce(new_center, op=dist.ReduceOp.SUM)
            new_center /= dist.get_world_size()
            initial_state[index] = new_center
            #tensor([0.4598, 0.6987], device='cuda:0') ,tensor([0.4598, 0.6987], device='cuda:1')

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))
        iteration = iteration + 1
        if rank==0:
           print(iteration)
           print(center_shift)
        # update tqdm meter
        # tqdm_meter.set_postfix(
        #     iteration=f'{iteration}',
        #     center_shift=f'{center_shift ** 2:0.6f}',
        #     tol=f'{tol:0.6f}'
        # )
        # tqdm_meter.update()
        if  iteration>iteration_:
            break

    return  initial_state
