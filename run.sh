#!/bin/bash
#sh run_pretrain_megatron_qwen.sh dsw ../../ 7B 1 8 1e-5 1e-6 2048 2048 293 bf16 1 1 sel true true true false 100000 /mnt/qwen-datasets/wudao_llamabpe_text_document /mnt/qwen-ckpts/Qwen-2-7b-hf-to-mg-tp1-pp1/ 10000000000 100000000 /mnt/output_patch_test
set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=8


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
batchsize=$1
lr=$2
output=$3
codebook=$4
epochs=$5
megatron_options=" \
    --batch-size ${batchsize} \
    --lr ${lr} \
    --codebook ${codebook} \
    --epochs ${epochs} \
    --output ${output} \
    "
        

run_cmd="torchrun $DISTRIBUTED_ARGS /mnt/data/user/lidehu/vae/ALIP/train.py
 ${megatron_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
