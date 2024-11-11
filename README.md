# DiversifyAnalysis
## Overview
This repo reproduces the experiments from paper "[OUT-OF-DISTRIBUTION REPRESENTATION LEARNING FOR TIME SERIES CLASSIFICATION](https://openreview.net/pdf?id=gUZWOE42l6Q)"

## How-to
### 1. preprocess
Download and prepare your datasets by:
``` python 
python prepare_dataset.py --data_dir ./data/ --dataset EMG
```
if executed successfully, dataset EMG will be saved to ./data/emg

### 2. train
```python 
python train.py --data_dir ./data/ --task cross_people --test_envs 0 --dataset emg --algorithm diversify --latent_domain_num 10 --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 5 --lr 0.01 --output ./data/train_output/act/cross_people-emg-Diversify-0-10-1-1-0-3-50-0.01 --output_model ./data/model_output
```
- specify your dataset with **--data_dir**
- specify your dataset name with **--dataset**
- training log will be saved to **--output**
- generated models will be saved to **--output_model**

### 3. evaluate

## References
```
@inproceedings{lu2022out,
  title={Out-of-distribution Representation Learning for Time Series Classification},
  author={Lu, Wang and Wang, Jindong and Sun, Xinwei and Chen, Yiqiang and Xie, Xing},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
