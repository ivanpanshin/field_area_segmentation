## Define model weights
EFFNET_FOLD_0_PATH="final_weights/effnet_v2_s_fold0_test_pseudo_ssl4eo_pretrain.pt"
EFFNET_FOLD_1_PATH="final_weights/effnet_v2_s_fold1_test_pseudo_ssl4eo_pretrain.pt"
EFFNET_FOLD_2_PATH="final_weights/effnet_v2_s_fold2_test_pseudo_ssl4eo_pretrain.pt"

MAXVIT_FOLD_0_PATH="final_weights/maxvit_fold0_test_pseudo_ssl4eo_pretrain.pt"
MAXVIT_FOLD_1_PATH="final_weights/maxvit_fold1_test_pseudo_ssl4eo_pretrain.pt"
MAXVIT_FOLD_2_PATH="final_weights/maxvit_fold2_test_pseudo_ssl4eo_pretrain.pt"

SERESNEXT_FOLD_0_PATH="final_weights/seresnext_fold0_test_pseudo_ssl4eo_pretrain.pt"
SERESNEXT_FOLD_1_PATH="final_weights/seresnext_fold1_test_pseudo_ssl4eo_pretrain.pt"
SERESNEXT_FOLD_2_PATH="final_weights/seresnext_fold2_test_pseudo_ssl4eo_pretrain.pt"

## Inference models
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=effnet_v2_s trainer.trainer_hyps.model_path=${EFFNET_FOLD_0_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/effnet_v2_fold_0.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=effnet_v2_s trainer.trainer_hyps.model_path=${EFFNET_FOLD_1_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/effnet_v2_fold_1.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=effnet_v2_s trainer.trainer_hyps.model_path=${EFFNET_FOLD_2_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/effnet_v2_fold_2.csv

torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=maxvit_base trainer.trainer_hyps.model_path=${MAXVIT_FOLD_0_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/maxvit_base_fold_0.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=maxvit_base trainer.trainer_hyps.model_path=${MAXVIT_FOLD_1_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/maxvit_base_fold_1.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=maxvit_base trainer.trainer_hyps.model_path=${MAXVIT_FOLD_2_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/maxvit_base_fold_2.csv

torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=seresnext trainer.trainer_hyps.model_path=${SERESNEXT_FOLD_0_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/seresnext_fold_0.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=seresnext trainer.trainer_hyps.model_path=${SERESNEXT_FOLD_1_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/seresnext_fold_1.csv
torchrun --nproc_per_node=1 --master-port=29500 mining_sites/test.py dataset=[test/competition_test] model=seresnext trainer.trainer_hyps.model_path=${SERESNEXT_FOLD_2_PATH} trainer.trainer_hyps.test_preds_output_path=inference/test_reproduced/seresnext_fold_2.csv

## Ensemble predictions
python mining_sites/ensemble.py subs_root=inference/test_reproduced output_path=inference/test_reproduced_final/test.csv