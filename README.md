# Field Area Segmentation

## Install 
1. Create a clean virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
pip install -v -e segmentation_models.pytorch
```

3. Download competition data and place it in `data` folder. In particular:
```
data/
└── images/
    └── test_*.tif
```

5. Download [weights](https://www.kaggle.com/datasets/ivanpan/final-weights) and place them in `final_weight`. In particular:
```
final_weights
└── unet_steroids_maxvit_1024_1024_model.pt
```

## Reproduce inference
```
bash bash_scripts/reproduce_inference.sh
```

