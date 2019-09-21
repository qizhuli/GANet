#! /bin/bash

python -W ignore::UserWarning \
                  predict.py --crop_height=1056 \
                  --crop_width=2064 \
                  --max_disp=192 \
                  --data_path='data/Cityscapes' \
                  --test_list='lists/cityscapes_val.list' \
                  --save_path='./result/cityscapes/val' \
                  --cityscapes=1 \
                  --resume='./checkpoint/kitti2015_final.pth'
