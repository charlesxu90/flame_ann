nohup python flame2_ann1_cuda0.py --data_dir ../data/ --epochs 20000 --device cuda 2>&1 >flame2_ann1.out &
nohup python flame2_ann2_cuda1.py --data_dir ../data/ --epochs 20000 --device cuda:1 2>&1 >flame2_ann2.out &
nohup python flame2_ann3_cpu.py --data_dir ../data/ --epochs 20000 --device cpu 2>&1 >flame2_ann3.out &


