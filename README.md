# Mining Sites

## Install 
1. Create a clean virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.dev.txt
pip install -r requirements.txt
```

3. Download competition data and place it in `data` folder. In particular:
```
data/
└── annotations
    └── answer.csv
└── images/
    ├── competition_data/
    │   └── train_*.tif
    └── competition_test/
        └── evaluation_*.tif
```

4. Download [SSL4EO](https://mediatum.ub.tum.de/1702379) dataset (only the `RGB` version) and place it in `data`. In particular:
```
data/
└── ssl4eo_rgb/
    ├── ***/
        └── ***/
            └── *.tif    
```

5. Download [weights](https://www.kaggle.com/datasets/ivanpan/final-weights) and place them in `final_weight`. In particular:
```
final_weights
└── effnet_***.pt
└── maxvit_***.pt
└── seresnext_***.pt
```

## Reproduce inference
```
bash bash_scripts/reproduce_inference.sh
```

## Reproduce training
0. Split data into folds
```
python mining_sites/preprocessing/create_folds.py
```

1. Train 1 stage models (train data of competition)
```
bash bash_scripts/train_effnet_v2_s_1stage.sh
bash bash_scripts/train_maxvit_base_1stage.sh
bash bash_scripts/train_seresnext_1stage.sh
```

2. Calculate pseudo for the competition test dataset
```
bash bash_scripts/test_effnet_v2_s_1stage.sh
bash bash_scripts/test_maxvit_base_1stage.sh
bash bash_scripts/test_seresnext_1stage.sh
```

Don't forget to insert paths to model weights in the corresponding bash files. 

3. Ensemble test predictions to create pseudo labels
```
python mining_sites/create_pseudo.py
```

4. Train 2 stage models (train + test data of competition)
```
bash bash_scripts/train_effnet_v2_s_2stage.sh
bash bash_scripts/train_maxvit_base_2stage.sh
bash bash_scripts/train_seresnext_2stage.sh
```

5. Preprocess SSL4EO dataset
```
python mining_sites/preprocessing/calculate_median_image.py
```

6. Calculate pseudo for the SSL4EO dataset
```
bash bash_scripts/test_effnet_v2_s_2stage.sh
bash bash_scripts/test_maxvit_base_2stage.sh
bash bash_scripts/test_seresnext_2stage.sh
```

7. Ensemble SSL4EO predictions to create pseudo labels
```
python mining_sites/create_pseudo.py subs_root=inference/ssl4eo output_path=inference/ssl4eo_pseudo/test.csv
```

8. Train 3 stage models (SS4LEO)
```
bash bash_scripts/train_effnet_v2_s_3stage.sh
bash bash_scripts/train_maxvit_base_3stage.sh
bash bash_scripts/train_seresnext_3stage.sh
```

9. Train 4 stage models (train + test data of competition with SS4LEO pretrain)
```
bash bash_scripts/train_effnet_v2_s_4stage.sh
bash bash_scripts/train_maxvit_base_4stage.sh
bash bash_scripts/train_seresnext_4stage.sh
```

10. Calculate test predictions for final sub
```
bash bash_scripts/test_effnet_v2_s_4stage.sh
bash bash_scripts/test_maxvit_base_4stage.sh
bash bash_scripts/test_seresnext_4stage.sh
```

11. Ensemble test predictions
```
python mining_sites/ensemble.py
```


