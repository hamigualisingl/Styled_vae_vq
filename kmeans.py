from sklearn.cluster import MiniBatchKMeans
import numpy as np
import torch
from k_means_ import kmeans, kmeans_dist, kmeans_predict
import os
import argparse
from torch import distributed, optim
parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=1000, type=int)
parser.add_argument("--n_class", default=1000, type=int)
parser.add_argument("--k", default=100, type=int)
parser.add_argument("--downsample", default=4, type=int)
parser.add_argument("--imagenet_feature_path", default="", type=str)
parser.add_argument("--save_dir", default="clustering_centers", type=str)
args = parser.parse_args()
######分布式
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
# print(world_size)是所有卡总和
distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
save_path = '/mnt/data/user/lidehu/vae/ALIP/kmeans'
os.makedirs(save_path, exist_ok=True)
feature_file=f"/mnt/data/user/lidehu/vae/ALIP/merge/merged_features_{rank}.pt"
print(feature_file)
map_location = torch.device(f'cuda:{local_rank}')
features =torch.load(feature_file, map_location=map_location)
        
features = features.reshape(-1, 128)

k=194560
print("开始啦")
center  =kmeans_dist(X=features, num_clusters=k, device=map_location,rank=rank)
print(center.shape)
if rank==0:
   torch.save(center, os.path.join(save_path, "k_means.pt"))
