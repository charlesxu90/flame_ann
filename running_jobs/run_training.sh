#!/bin/bash
nohup python flame2_ann1_cuda0.py --data_dir ../data/ --epochs 20000 --device cuda --ckpt ann1_models/1/model_1627_21.8412.pt 2>&1 >flame2_ann1.out &
nohup python flame2_ann2_cuda1.py --data_dir ../data/ --epochs 20000 --device cuda:1 --ckpt ann2_models/2/model_1070_0.0009.pt 2>&1 >flame2_ann2.out &
nohup python flame2_ann3_cpu.py --data_dir ../data/ --epochs 20000 --device cpu --ckpt ann3_models/1/model_2114_0.9049.pt 2>&1 >flame2_ann3.out &
