FOLD_0_PATH="maxvit_base_fold0/epoch_1.pt"
FOLD_1_PATH="maxvit_base_fold0/epoch_1.pt"
FOLD_2_PATH="maxvit_base_fold0/epoch_1.pt"

torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=maxvit_base trainer.trainer_hyps.model_path=${FOLD_0_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_final/maxvit_base_fold_0.csv transform=resize
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=maxvit_base trainer.trainer_hyps.model_path=${FOLD_1_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_final/maxvit_base_fold_1.csv transform=resize
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=maxvit_base trainer.trainer_hyps.model_path=${FOLD_2_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_final/maxvit_base_fold_2.csv transform=resize