
![License](https://img.shields.io/github/license/senli1073/LNRL)
![LastCommit](https://img.shields.io/github/last-commit/senli1073/LNRL)
------------------

- [Architecture](#architecture)
- [Introduction](#introduction)
- [Usage](#usage)
  - [Data preparation](#data-preparation)
  - [Training](#training)
  - [Testing](#testing)
- [License](#license)

## Architecture
<p align="center">
  <img src="https://raw.githubusercontent.com/senli1073/LNRL/main/images/Architecture.png">
</p>

## Introduction
Label Noise-Robust Learning (LNRL) approach was designed for handling label noise in microseismic tasks with small-scale datasets. LNRL aligns feature representation and label representation distribution in multiple feature spaces, learns the correlation between instances and label noise, and mitigates the impact of label noise.

The code of this project is developed based on [SeisT](https://github.com/senli1073/SeisT). 

## Usage

### Data Preparation

- **For training and evaluation**
  
  Create a new file named `mydata.py` in the directory `dataset/` to read the metadata and seismograms of the dataset. And the `@register_dataset` decorator needs to be used to register the custom dataset. 

  (Please refer to the code example `datasets/sos.py`)

### Training

- **Model**<br/>
  Before starting training, please make sure that the model code is in the directory `models/` and register it using the `@register_model` decorator. All available models in the project can be inspected by using the following method: 
  ```Python
  >>> from models import get_model_list
  >>> get_model_list()
  ['lnrl','seist']
  ```

- **Model Configuration**<br/>
  The configurations of the loss functions, labels, and the corresponding models are in `config.py` which also provides a detailed explanation of all the fields.


- **Start training**<br/>
  To start training with a CPU or a single GPU, please use the following command to start training:
  ```Shell
  python main.py \
    --seed 0 \
    --mode "train_test" \
    --model-name "lnrl" \
    --log-base "./logs" \
    --device "cuda:0" \
    --data "/root/data/Datasets/SOS" \
    --dataset-name "sos" \
    --sigma 600 \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --shuffle true \
    --workers 8 \
    --in-samples 6000 \
    --augmentation true \
    --epochs 200 \
    --patience 30 \
    --batch-size 300
  ```
  
  Use `torchrun` if training with multiple GPUs.

  There are also a variety of other custom arguments which are not mentioned above. Use the command `python main.py --help` to see more details.


### Testing
  Use the following command to start testing:

  ```Shell
  python main.py \
    --seed 0 \
    --mode "test" \
    --model-name "lnrl" \
    --log-base "./logs" \
    --device "cuda:0" \
    --data "/root/data/Datasets/SOS" \
    --dataset-name "sos" \
    --data-split true \
    --train-size 0.8 \
    --val-size 0.1 \
    --workers 8 \
    --in-samples 6000 \
    --batch-size 300
  ```

  It should be noted that the `train_size`, `val_size`, and `seed` in the test phase must be consistent with that training phase. Otherwise, the test results may be distorted.


## Acknowledgement
This project refers to some excellent open source projects: [PhaseNet](https://github.com/AI4EPS/PhaseNet), [EQTransformer](https://github.com/smousavi05/EQTransformer)


## License
Copyright S.Li et al. 2024. Licensed under an MIT license.







