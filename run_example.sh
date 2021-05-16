#!/bin/bash

python3 train_trades_cifar10_nodist.py --batch-size 128 --epochs 20 \
        --model-dir model_logs/adv_train_comp_5012_fixed/ \
        --log_filename ./result/logfile_train_comp_5012_fixed.csv \
        --order fixed --enable 5,0,1,2 \
        --dist comp \
        --model_path model_logs/adv_train_comp_5012_fixed/model_best.pth
