#!/bin/bash

python3 train_trades_cifar10_nodist.py --batch-size 128 --epochs 10 \
        --model-dir model_logs/adv_train_comp_012345_fixed/ \
        --log_filename ./result/logfile_train_comp_012345_fixed.csv \
        --order fixed --enable 0,1,2,3,4,5 \
        --dist comp
