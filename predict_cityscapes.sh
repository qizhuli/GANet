#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=${NGPUS} \
    predict.py --crop_height=1056 \
    --crop_width=2064 \
    --max_disp=192 \
    --data_path='data/Cityscapes' \
    --test_list='lists/cityscapes_train.list' \
    --save_path='./result/cityscapes/train' \
    --cityscapes=1 \
    --resume='./checkpoint/kitti2015_final.pth'

