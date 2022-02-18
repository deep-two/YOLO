# YOLO

pytorch 1.9.1

## 1. Introduction
**Faster-Rcnn only with Pytorch.** No need to build.
We use [jwyang's repo](https://github.com/jwyang/faster-rcnn.pytorch) as reference, so this repo has similar structure and some code come from jwyang's repo. 

## 2. Preparation
### 2.1 Prerequisites
Python 3.8
Pytorch 1.9.0+cu111
torchvision 0.10.0+cu111
numpy 1.19.5

```bash
pip install -r requirements.txt
```

### 2.2 Code-Preparing
```bash
git clone https://github.com/deep-two/YOLO.git
```

### 2.3 Data Preparation
Pascal VOC
1. Download the training, validation, test data and annotations
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

2. Extract all of these tars into one directory named COCODevKit
 ```bash
 tar -xvf VOCtrainval_11-May-2012.tar
 ```
3. Data dir should like this
```
   VOCDevKit
        |-- VOC2012
                |-- Annotations
                        |-- [xxxxxxxxxxxx].xml
                |-- ImageSets
                        |-- Action
                        |-- Layout
                        |-- Main
                        |-- Segmentation
                |-- JPEGImages
                        |-- [xxxxxxxxxxxx].jpg
```

## 3. Train
```bash
python train.py --cuda --batch_size $BATCH_SIZE \
            --dataset $DATASET_PATH
```
