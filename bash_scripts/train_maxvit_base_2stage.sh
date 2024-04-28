torchrun --nproc_per_node=1 --master-port=29500 mining_sites/train.py dataset=[train/fold0_pseudo,val/fold0] scheduler.scheduler.T_max=20000 model=maxvit_base logging.logging_dir=maxvit_base_fold0_stage2 dataset.train.train.multiplier=10
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/train.py dataset=[train/fold1_pseudo,val/fold1] scheduler.scheduler.T_max=20000 model=maxvit_base logging.logging_dir=maxvit_base_fold1_stage2 dataset.train.train.multiplier=10
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/train.py dataset=[train/fold2_pseudo,val/fold2] scheduler.scheduler.T_max=20000 model=maxvit_base logging.logging_dir=maxvit_base_fold2_stage2 dataset.train.train.multiplier=10