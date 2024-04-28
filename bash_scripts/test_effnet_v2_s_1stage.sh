FOLD_0_PATH="effnet_v2_s/epoch_9.pt"
FOLD_1_PATH="effnet_v2_s/epoch_9.pt"
FOLD_2_PATH="effnet_v2_s/epoch_9.pt"

torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=effnet_v2_s trainer.trainer_hyps.model_path=${FOLD_0_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test/effnet_v2_fold_0.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=effnet_v2_s trainer.trainer_hyps.model_path=${FOLD_1_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test/effnet_v2_fold_1.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=effnet_v2_s trainer.trainer_hyps.model_path=${FOLD_2_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test/effnet_v2_fold_2.csv