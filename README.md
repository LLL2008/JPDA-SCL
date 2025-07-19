<div align="center">
    <h2>
        Enhancing Domain Generalization in Hyperspectral Image Classification via Joint-Product Distribution Alignment and Supervised Contrastive Learning
    </h2>
</div>
<br>

<div align="center">
  <img src="./logs/JPDA-SCL.png" width="800"/>
</div>
<br>

## Introduction

This repository is the code implementation of the paper [Enhancing Domain Generalization in Hyperspectral Image Classification via Joint-Product Distribution Alignment and Supervised Contrastive Learning], which is based on the [MDGTnet](https://github.com/Cherrieqi/MDGTnet) project.

## Installation

## Requirements

This code is based on **Python 3.10.14** and **Pytorch 2.7.1**.

You can install all dependencies via:

``bash
pip install -r requirements.txt

## Models

**· SD--H13+H18 :[model\_.pth](https://pan.baidu.com/s/1Dmj8fSZjNHA5Ay_-ZctW-w?pwd=1234)**

**· SD--PU+PC :[model\_.pth](https://pan.baidu.com/s/1YRl9o7SqiivkBxhb6MkFJg?pwd=1234)**

## Datasets

**[raw](https://pan.baidu.com/s/15HsBrk2YlkrP8PfsfTFukA?pwd=1234) :** Houston2013 / Houston2018 / PaviaU / PaviaC

**[H13+H18 -- PU/PC](https://pan.baidu.com/s/1h6O0lagIPE57ZojxM60xWQ?pwd=1234) :** gen_H13 / gen_H18 / gen_PU / gen_PC

**[PU+PC--H13/H18](https://pan.baidu.com/s/1k2wWVf2KaP4m3zsRcwOqGQ?pwd=1234)** **:** gen_H13 / gen_H18 / gen_PU / gen_PC

## Getting start:

##### · Dataset structure
```
```
data/SD_H1318
├── gen_H13
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy
├── gen_H18
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy
├── gen_PC
│   ├── img_norm_all.npy
│   └── gt_norm_all.npy
└── gen_PU
     ├── img_norm_all.npy
     └── gt_norm_all.npy
```
```
data/SD_PUPC
├── gen_H13
│ ├── img_norm_all.npy
│ └── gt_norm_all.npy
├── gen_H18
│ ├── img_norm_all.npy
│ └── gt_norm_all.npy
├── gen_PC
│ ├── img_norm_all.npy
│ └── gt_norm_all.npy
└── gen_PU
├── img_norm_all.npy
└── gt_norm_all.npy
```

```
data/raw
├── Houston2013
│   ├── Houston.mat
│   └── Houston_gt.mat
├── Houston2018
│   ├── HoustonU.mat
│   └── HoustonU_gt.mat
├── PaviaC
│   ├── pavia.mat
│   └── pavia_gt.mat
└── PaviaU
     ├── paviaU.mat
     └── paviaU_gt.mat


**NOTE:**

​Training and test data can be generated via _data_pre_SD_xxxxx.py_ respectively. Where _\_H13H18_ indicates that the source domains are H13 and H18.

##### Train

​python train_SD_H13H18.py.

##### Test

​python test_TD_PUPC.py.

## Contact Us

If you have any other questions, please contact us in time.
