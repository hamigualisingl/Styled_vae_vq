import argparse
import logging
import math
import os
import sys
import time
import torch
from dataset import get_dataloader
import torch.nn as nn
from torch import distributed, optim
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist

import dnnlib

from model import  VQVAE_Transformer_vit_sd3_hug_4096#VQVAE_Transformer_expert_vit_no_qunat_hug_32768
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--beta1", type=float, default=0.9, help="adamw")
parser.add_argument("--beta2", type=float, default=0.98, help="adamw")
parser.add_argument("--epochs", type=int, default=128)
parser.add_argument("--gradient-acc", type=int, default=1)#load
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--lr-scheduler", default="cosine")
parser.add_argument("--input-size", default=256, type=int)
parser.add_argument("--codebook", default=1024, type=int)
parser.add_argument("--load", default=0, type=int)
parser.add_argument("--exctiqunat", type=int, default=0)
parser.add_argument("--modelname", type=int, default=0)
parser.add_argument("--local-loss",default=False,help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)")
parser.add_argument("--gather-with-grad",default=False,help="enable full distributed gradient for feature gather")
parser.add_argument("--horovod",default=False,action="store_true",help="Use horovod for distributed training.")
parser.add_argument("--optimizer", default="adamw")
parser.add_argument("--output", default="/mnt/data/user/lidehu/vae/ALIP/feature")
parser.add_argument("--train-num-samples", type=int, default=10640526)
parser.add_argument("--weight-decay", type=float, default=0.2, help="Weight decay.")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
def get_state_dict(model_weight):
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    return state_dict_removed

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def main():
    os.makedirs(args.output, exist_ok=True)
   
    if rank == 0:
        summary_writer = SummaryWriter(os.path.join(args.output, "tensorboard"))
    else:
        summary_writer = None
    #model_alip = VQVAE_Transformer_vit_sd3_hug_4096(width=1024, layers=24, heads=16, mlp_ratio=4.0,decoder_dim=512)
    model_alip = VQVAE_Transformer_vit_sd3_hug_4096(width=1024, layers=24, heads=16, mlp_ratio=4.0,emb_dim=128)
    model_alip.cuda()
    model_alip = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_alip)
    model_alip = torch.nn.parallel.DistributedDataParallel(
        module=model_alip,
        bucket_cap_mb=32,
        find_unused_parameters=True,
        static_graph=True)
    #加载模型权重，指定加载到设备，不然都跑到0卡去了
    map_location = torch.device(f'cuda:{local_rank}')
    model_alip.load_state_dict(
    torch.load("/mnt/data/user/lidehu/vae/ALIP/out_put_1060w_noise0.0005_l2/model_11.pt", map_location=map_location))
    #torch.save(obj=model_alip.state_dict(), f=os.path.join(args.output, "model_{}.pt".format(str(rank))))
     ######测试 不同卡权重是否一致,结论:致     
    # train_loader = dali_dataloader(args)
    dataloader, sampler = get_dataloader(type='CC3M',
                                ######其实是yffcc15m数据集
                            batch_size=args.batch_size ,
                            img_shape=(256, 256),
                            num_workers=16,
                            dist_train=True,
                            use_lmdb=False)
    print("完成数据加载的构建")

    #提前下载好

    steps_per_epoch = args.train_num_samples // world_size // args.batch_size + 1
    steps_total = int(args.epochs * steps_per_epoch)
    print(steps_total)
    torch.distributed.barrier()
  
    torch.distributed.barrier()
     
    if 1==1:
        sampler.set_epoch(0)
        for i, img in enumerate(dataloader):
            img = img.cuda()
            images_feature,_,_=model_alip(img)# #((36,bs,128)) 
            #images_feature_cpu = images_feature.cpu()
            file_name = os.path.join(args.output, f"img_feature_batch_{i}_rank_{rank}.pt")
            torch.save(images_feature, file_name)

           

             

if __name__ == "__main__":
    main()
