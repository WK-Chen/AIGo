export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=7 nohup python run.py > run.log 2>&1 &