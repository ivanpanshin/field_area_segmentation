python field_area_segmentation/preprocessing/preprocess_test_dataset.py
python field_area_segmentation/preprocessing/create_crops.py

torchrun --nproc_per_node=1 --master-port=29500 field_area_segmentation/test.py