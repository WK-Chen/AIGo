export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=4 nohup python run.py --round=0 --target_round=10 > run.log 2>&1 &