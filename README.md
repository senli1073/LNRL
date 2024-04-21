
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
We introduce a Label Noise Robust Learning (LNRL) method for handling label noise in microseismic tasks with small-scale datasets. LNRL aligns feature representation and label representation distribution in multiple feature spaces, learns the correlation between instances and label noise, and mitigates the impact of label noise.

The code of this project is modified based on [SeisT](https://github.com/senli1073/SeisT). 

## Usage

### Data Preparation

- **For training and evaluation**
  
  Create a new file named `yourdata.py` in the directory `dataset/` to read the metadata and seismograms of the dataset. And you need to use `@register_dataset` decorator to register your dataset. 

  (Please refer to the code samples `datasets/sos.py`)

### Training

- **Model**<br/>
  Before starting training, please make sure that your model code is in the directory `models/` and register it using the `@register_model` decorator. You can inspect the models available in the project using the following method: 
  ```Python
  >>> from models import get_model_list
  >>> get_model_list()
  ['seist','lnrl']
  ```

- **Model Configuration**<br/>
  The configuration of the loss function and model labels is in `config.py`, and a more detailed explanation is provided in this file.


- **Start training**<br/>
  If you are training with a CPU or a single GPU, please use the following command to start training:
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
  
  If you are training with multiple GPUs, please use `torchrun` to start training:
  ```Shell
  torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    main.py \
      --seed 0 \
      --mode "train_test" \
      --model-name "lnrl" \
      --log-base "./logs" \
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
  
  There are also many other custom arguments, see `main.py` for more details.


### Testing
  If you are testing with a CPU or a single GPU, please use the following command to start testing:

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
  
  If you are testing with multiple GPUs, please use `torchrun` to start testing:
  ```Shell
  torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    main.py \
      --seed 0 \
      --mode "test" \
      --model-name "lnrl" \
      --log-base "./logs" \
      --data "/root/data/Datasets/SOS" \
      --dataset-name "sos" \
      --data-split true \
      --train-size 0.8 \
      --val-size 0.1 \
      --workers 8 \
      --in-samples 6000 \
      --batch-size 300
  ```

  It should be noted that the `train_size` and `val_size` during testing must be consistent with that during training, and the `seed` must be consistent. Otherwise, the test results may be distorted.


## License
Copyright S.Li et al. 2024. Licensed under an MIT license.







