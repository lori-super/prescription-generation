# MMPG

- This project is for the paper “A novel herbal compatibility-aware approach for TCM prescription generation'' 
- For more details, please refer to our paper ”A novel herbal compatibility-aware approach for TCM prescription generation“

## Installation

- PyTorch needs to be installed first (preferably inside a conda environment). Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform

## Experiments

### Train models

- Refer to [train](https://github.com/lori-super/prescription-generation/tree/main/train)t , all our training details are in [train.py](https://github.com/lori-super/prescription-generation/blob/main/train/train.py) and [train_iter.py](https://github.com/lori-super/prescription-generation/blob/main/train/train_iter.py)
- Refer to [models](https://github.com/lori-super/prescription-generation/tree/main/models). all our model details are in [model.py](https://github.com/lori-super/prescription-generation/blob/main/models/model.py)
- Refer to [main.py](https://github.com/lori-super/prescription-generation/blob/main/main.py) to start the model please use this file

### Predict

- Refer to [predict.py](https://github.com/lori-super/prescription-generation/blob/main/predict.py) to get all the functions developed to predict the prescription using the trained model

## Data

- Due to our annotation work on the data, the annotated data is a private data set, if you want to obtain the data, please contact the relevant author
- The pre-markup dataset is available via the following link:
  - CS-seq2seq:https://github.com/lancopku/tcm_prescription_generation/tree/master
  - PTM:https://github.com/yao8839836/PTM

## Reference

```
@article{rechao},
title = {A novel herbal compatibility-aware approach for TCM prescription generation},
author = {},
journal = {},
year = {},
publisher={},
doi = {},
url = {}
}
#No published paper, will be updated later
```

