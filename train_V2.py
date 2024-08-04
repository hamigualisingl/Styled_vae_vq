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

from modelV2 import  VQVAE_Transformer_vit_sd3_hug_4096#VQVAE_Transformer_expert_vit_no_qunat_hug_32768
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
parser.add_argument("--output", default="/mnt/data/user/lidehu/vae/ALIP/out_put_test")
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
    map_location = torch.device(f'cuda:{local_rank}')
    os.makedirs(args.output, exist_ok=True)
    init_logging(rank, args.output)
    if rank == 0:
        summary_writer = SummaryWriter(os.path.join(args.output, "tensorboard"))
    else:
        summary_writer = None
    model_alip = VQVAE_Transformer_vit_sd3_hug_4096(width=1024, layers=24, heads=16, mlp_ratio=4.0,emb_dim=128)
    model_alip.cuda()
    model_alip = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_alip)
    # static =torch.load('/mnt/data/user/lidehu/vae/ALIP/test.pt', map_location=map_location)
    # model_alip.load_state_dict(static, strict=True)
    model_alip = torch.nn.parallel.DistributedDataParallel(
        module=model_alip,
        bucket_cap_mb=32,
        find_unused_parameters=True,
        static_graph=True)
    #加载模型权重  需要指定加载设备，不然会出问题，因为版是0卡保存的，到时候都是加载到0卡
    dataloader, sampler = get_dataloader(type='CC3M',
                                ######其实是yffcc15m数据集
                            batch_size=args.batch_size ,
                            img_shape=(256, 256),
                            num_workers=16,
                            dist_train=True,
                            use_lmdb=False)
    print("完成数据加载的构建")
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().cuda()#####第二阶段就去掉这个
    #提前下载好
    model_path = '/mnt/data/user/lidehu/vae/ALIP/vgg16.pt'
    with dnnlib.util.open_url(model_path) as f:
        vgg16 = torch.jit.load(f).eval().cuda()
    #提前下载好
    global_step = GlobalStep()
    steps_per_epoch = args.train_num_samples // world_size // args.batch_size + 1
    steps_total = int(args.epochs * steps_per_epoch)
    print(steps_total)
    torch.distributed.barrier()
    opt = torch.optim.AdamW(
        params=[{"params": model_alip.parameters()}],#第一阶段的
        lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
     
    if args.lr_scheduler == "cosine":
        assert isinstance(args.epochs, int)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=[args.lr],
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.1,
        )

    elif args.lr_scheduler == "linear":
        lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=opt, start_factor=1.0, end_factor=0.0,
            total_iters=int(args.epochs * steps_per_epoch))
    else:
        raise
    
    callback_func = SpeedCallBack(5, steps_total, args.batch_size)
    auto_scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=200)
    start_epoch = 0
    reconstruction_criterionl2 = nn.MSELoss()  
    reconstruction_criterionl1 = nn.L1Loss() #####继续学习
    model_alip.train()
    torch.distributed.barrier()
     
    for epoch in range(start_epoch, math.ceil(args.epochs)):
        sampler.set_epoch(epoch)
        for _, img in enumerate(dataloader):
            img = img.cuda()
           #((12,3,256,256))
            reconstructed_images,kl_loss=model_alip(img)#
            reconstruction_lossl2 = reconstruction_criterionl2(img, reconstructed_images.float())
            img_features = vgg16((img + 1) * (255/2), resize_images=False, return_lpips=True)
            reconstructed_images_features = vgg16((reconstructed_images.float() + 1) * (255/2), resize_images=False, return_lpips=True)
            perc_loss =(img_features - reconstructed_images_features).square().sum(1).mean()
        ########感知损失使用L2_los,重建损失使用l1_loss
            loss =  perc_loss+reconstruction_lossl2#
            auto_scaler.scale(loss).backward()
            if global_step.step % args.gradient_acc == 0:
                auto_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model_alip.parameters(), 1)
                auto_scaler.step(opt)
                auto_scaler.update()
                opt.zero_grad()
            lr_scheduler.step()
            global_step.step += 1

            with torch.no_grad():
                callback_func( lr_scheduler,float(reconstruction_lossl2), float(perc_loss), float(0), global_step.step)
                if summary_writer is not None:
                    #summary_writer.add_scalar(tag="unhit_count", scalar_value=unhit_count, global_step=global_step.step)
                    summary_writer.add_scalar(tag="lossl2", scalar_value=reconstruction_lossl2.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="percent_loss", scalar_value=perc_loss.item(), global_step=global_step.step)
                    #summary_writer.add_scalar(tag="reconstrution_loss", scalar_value=reconstruction_lossl1.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="lr_backbone",
                                             scalar_value=lr_scheduler.get_last_lr()[0],
                                             global_step=global_step.step)
              
            # if global_step.step > steps_total:
            #     break

      
        if rank == 0 : 
            torch.save(obj=model_alip.state_dict(), f=os.path.join(args.output, "model_{}.pt".format(str(epoch))))
#            f=os.path.join(args.output, "lr_model_{}".format(str(epoch)))
#             state = {
#            'epoch': epoch,
#            'optimizer': opt.state_dict(),
#     'scheduler': lr_scheduler.state_dict() ,
#     'model_state_dict': model_alip.state_dict()
# }
            #torch.save(state, f)
    if summary_writer is not None:
        summary_writer.close()


def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GlobalStep:
    def __init__(self, step: int = 0):
        self.step = int(step)

    def update(self):
        self.step += 1


class SpeedCallBack(object):
    def __init__(self, frequent, steps_total, batch_size):
        self.batch_size = batch_size
        self.frequent = frequent
        self.steps_total = steps_total
        self.loss_metric = AverageMeter()
        self.perc_loss_metric = AverageMeter()
        self.q_loss_metric = AverageMeter()
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.time_start = time.time()
        self.init = False
        self.tic = 0

    def __call__(
            self,
            lr_scheduler: optim.lr_scheduler._LRScheduler,
            loss,
            perc_loss,
            q_loss,
            global_step
            ):#clip_loss,
        assert isinstance(loss, float)

        self.loss_metric.update(loss)
        self.perc_loss_metric.update(perc_loss)
        self.q_loss_metric.update(q_loss)

        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = (self.frequent * self.batch_size / (time.time() - self.tic))
                    self.tic = time.time()
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")

                loss_metric_format = f"[{self.loss_metric.avg :.3f}]"
                self.loss_metric.reset()
                perc_loss_metric_format = f"[{self.perc_loss_metric.avg :.3f}]"
                self.perc_loss_metric.reset()
                q_loss_metric_format = f"[{self.q_loss_metric.avg :.3f}]"
                self.q_loss_metric.reset()


                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.steps_total)
                time_for_end = time_total - time_now
                lr_1 = lr_scheduler.get_last_lr()[0]
                msg = f"rank:{int(speed) :d} "
                msg += f"total:{int(speed_total) :d} "
                msg += f"lr:[{lr_1 :.8f}] "
                msg += f"step:{global_step :d} "
                msg += f"required:{time_for_end :.1f} hours "
                msg += f"reconstr_loss:{loss_metric_format} "
                msg += f"enty_loss:{perc_loss_metric_format} "
                msg += f"q_loss:{q_loss_metric_format} "

                if self.rank == 0:
                    logging.info(msg)
            else:
                self.init = True
                self.tic = time.time()
                

if __name__ == "__main__":
    main()

                

if __name__ == "__main__":
    main()
