#!/bin/bash

#python3 train_trades_cifar10_nodist.py --batch-size 128 --epochs 20 \
#        --model-dir ./adv_train_comp_hue/ \
#        --log_filename ./result/logfile_train_comp_hue.csv \
#        --order fixed --enable 0 \
#        --dist comp


python3 train_trades_cifar10_nodist.py --batch-size 128 --epochs 20 \
        --model-dir model_logs/adv_train_comp_012_random \
        --log_filename ./result/adv_train_comp_012_random.csv \
        --order random --enable 0,1,2 \
        --dist comp \
        --model_path model_logs/adv_train_comp_012_random/model_best.pth

