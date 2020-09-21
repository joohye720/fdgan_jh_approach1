![Python 3](https://img.shields.io/badge/python-3-green.svg) ![Pytorch 0.3](https://img.shields.io/badge/pytorch-0.3-blue.svg)
# RGB-IR Cross-Modality Person Re-Identification with Pose-transferred Image Generation

<p align="center"><img src='framework.jpg' width="600px"></p>

[JOO-HYE PARK](Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology (KAIST), Daejeon 34141, South Korea (e-mail: weetweet720@gmail.com))\, [DONG-SOO KWON](Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology (KAIST), Daejeon 34141, South Korea (e-mail: kwonds@kaist.ac.kr)\

Pytorch implementation for our master theis research. With the two-generator based proposed approah :  we are able to learn **identity-related** and **modality and pose-unrelated** representations.

## Dataset
- SYSU-MM01 pedestrian multi modality public dataset
- It contains both RGB images and IR images per one identity

## Prerequisites
- Python 3
- [Pytorch](https://pytorch.org/) (We run the code under version 0.3.1, maybe lower versions also work.)

## Getting Started
## RGB to IR image translation 
- In two separate generators-based approach, there is an IR generator that converts the modality from RGB to IR.
- The pixel alignment module of AlignGAN was utilized for IR generator.

### Installation
- Install dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by:
```
pip install scipy, pillow, torchvision, sklearn, h5py, dominate, visdom
```
- Clone this repo:
```
git clone https://github.com/yxgeee/FD-GAN
cd FD-GAN/
```

### Datasets
We conduct experiments on [Market1501](http://www.liangzheng.org/Project/project_reid.html), [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation), [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) datasets. We need pose landmarks for each dataset during training, so we generate the pose files by [Realtime Multi-Person Pose Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation). And the raw datasets have been preprocessed by the code in [open-reid](https://github.com/Cysu/open-reid). 
Download the prepared datasets following below steps:
- Create directories for datasets:
```
mkdir datasets
cd datasets/
```
- Download these datasets through the links below, and `unzip` them in the same root path.  
*Market1501*: [[Google Drive]](https://drive.google.com/open?id=1LS5_bMqv-37F14FVuziK63gz0wPyb0Hh) [[Baidu Pan]](https://pan.baidu.com/s/13C7zcELYzd-5EnjAeDcX9A)  
*DukeMTMC-reID*: [[Google Drive]](https://drive.google.com/open?id=1Ujtm-Cq7lpyslBkG-rSBjkP1KVntrgSL) [[Baidu Pan]](https://pan.baidu.com/s/1B52P9RzTXv0JLmfmiA3aKg)  
*CUHK03*: [[Google Drive]](https://drive.google.com/open?id=1R7oCwyMHYIxpRVsYm7-2REmFopP9TSXL) [[Baidu Pan]](https://pan.baidu.com/s/1zH2jcEa_igC7Lzgts4FwVQ)  

## Usage
As mentioned in the original [paper](https://arxiv.org/abs/1810.02936), there are three stages for training our proposed framework.

### Stage I: reID baseline pretraining
We use a Siamese baseline structure based on `ResNet-50`. You can train the model with follow commands,
```
python baseline.py -b 256 -j 4 -d market1501 -a resnet50 --combine-trainval \
					--lr 0.01 --epochs 100 --step-size 40 --eval-step 5 \
					--logs-dir /path/to/save/checkpoints/
```
You can train it on specified GPUs by setting `CUDA_VISIBLE_DEVICES`, and change the dataset name `[market1501|dukemtmc|cuhk03]` after `-d` to train models on different datasets.  
Or you can download the pretrained baseline model directly following the link below,
- Market1501_baseline_model: [[Google Drive]](https://drive.google.com/open?id=1oNLf-gazgfN0EqkdIOKtcJSBx22BuO1-) [[Baidu Pan]](https://pan.baidu.com/s/1H0SNJmaM9GmYN2WE6W60Hw)
- DukeMTMC_baseline_model: [[Google Drive]](https://drive.google.com/open?id=1iVXIaXT6WQzKuLD3eDcBZB-3aNeZ6Ivf) [[Baidu Pan]](https://pan.baidu.com/s/1CCFjy9We7F9ZHpdTL65vxQ)
- CUHK03_baseline_model: [[Google Drive]](https://drive.google.com/open?id=1jubhvKl_Ny9b89wbX0-u2GhPEeXMLaUQ) [[Baidu Pan]](https://pan.baidu.com/s/1wtyfhiyxx6jWapMyR5x0Ig)

<a name="stageI"></a>And **test** them with follow commands,
```
python baseline.py -b 256 -d market1501 -a resnet50 --evaluate --resume /path/of/model_best.pth.tar
```

### Stage II: FD-GAN pretraining
We need to pretain FD-GAN with the image encoder part (*E* in the original paper and *net_E* in the code) fixed first. You can train the model with follow commands,
```
python train.py --display-port 6006 --display-id 1 \
	--stage 1 -d market1501 --name /directory/name/of/saving/checkpoints/ \
	--pose-aug gauss -b 256 -j 4 --niter 50 --niter-decay 50 --lr 0.001 --save-step 10 \
	--lambda-recon 100.0 --lambda-veri 0.0 --lambda-sp 10.0 --smooth-label \
	--netE-pretrain /path/of/model_best.pth.tar
```
You can train it on specified GPUs by setting `CUDA_VISIBLE_DEVICES`. For main arguments,
- `--display-port`: display port of [visdom](https://github.com/facebookresearch/visdom), e.g., you can visualize the results by `localhost:6006`.
- `--display-id`: set `0` to disable [visdom](https://github.com/facebookresearch/visdom).
- `--stage`: set `1` for Stage II, and set `2` for stage III.
- `--pose-aug`: choose from `[no|erase|gauss]` to make augmentations on pose maps.
- `--smooth-label`: smooth the label of GANloss or not. 

Other arguments can be viewed in [options.py](https://github.com/yxgeee/FD-GAN/blob/master/fdgan/options.py).
Also you can directly download the models for stage II,
- Market1501_stageII_model: [[Google Drive]](https://drive.google.com/open?id=1kIBuPzz-Ig70dE3rU-5-kyo3nGJP01NS) [[Baidu Pan]](https://pan.baidu.com/s/1X7T2yJPclZNzY4Bhr_wuuQ)
- DukeMTMC_stageII_model: [[Google Drive]](https://drive.google.com/open?id=1dD1cbg2jo5qhPbkMbsRYACRcVMrm28-o) [[Baidu Pan]](https://pan.baidu.com/s/17wELt0YdjTVzEbe_gRu60A)
- CUHK03_stageII_model: [[Google Drive]](https://drive.google.com/open?id=1552oDot-vgA27b-mCspJAuzaOl685koz) [[Baidu Pan]](https://pan.baidu.com/s/1pWmc8fNgC2xjDxM2Gb-pYA)

There are four models in each directory for separate nets.

**Notice**: 
If you use `visdom` for visualization by setting `--display-id 1`, you need to open a new window and run the script `python -m visdom.server -port=6006` before running the main program, where `-port` should be consistent with `--display-port`. 

### Stage III: Global finetuning
Finetune the whole framework by optimizing all parts. You can train the model with follow commands,
```
python train.py --display-port 6006 --display-id 1 \
	--stage 2 -d market1501 --name /directory/name/of/saving/checkpoints/ \
	--pose-aug gauss -b 256 -j 4 --niter 25 --niter-decay 25 --lr 0.0001 --save-step 10 --eval-step 5 \
	--lambda-recon 100.0 --lambda-veri 10.0 --lambda-sp 10.0 --smooth-label \
	--netE-pretrain /path/of/100_net_E.pth --netG-pretrain /path/of/100_net_G.pth \
	--netDi-pretrain /path/of/100_net_Di.pth --netDp-pretrain /path/of/100_net_Dp.pth
```
You can train it on specified GPUs by setting `CUDA_VISIBLE_DEVICES`.  
We trained this model on a setting of batchsize 256. If you don't have such or better hardware, you may decrease the batchsize (the performance may also drop).
Or you can directly download our final model,
- Market1501_stageIII_model: [[Google Drive]](https://drive.google.com/open?id=1w8xqopW0icA3VIxZyelI9k-Fb8rRCME7) [[Baidu Pan]](https://pan.baidu.com/s/1JE3Mwh0CxQ5EKkzLr7nEPg)
- DukeMTMC_stageIII_model: [[Google Drive]](https://drive.google.com/open?id=1axBHUcI7JmPbw8Y_mSpMKWIY9FUfFKMI) [[Baidu Pan]](https://pan.baidu.com/s/1tBF67qZrDmSgxOKENjUdFQ)
- CUHK03_stageIII_model: [[Google Drive]](https://drive.google.com/open?id=1q6HkDlDUIV9YNUwAggy-HI9zYQjt7Ihk) [[Baidu Pan]](https://pan.baidu.com/s/1fUaNTlOXjtEUQSq217X25Q)

And **test** `best_net_E.pth` by the same way as mentioned in [Stage I](#stageI).

## TODO
- generate IR images from RGB to IR image translation
- scripts for generate pose landmarks.
- generate specified images.


## Acknowledgements
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [open-reid](https://github.com/Cysu/open-reid) and FD-GAN(https://github.com/yxgeee/FD-GAN).
