## Modified SRGAN

  This repo was forked from @zsdonghao 's [tensorlayer/srgan](https://github.com/tensorlayer/srgan) repo.  
  Referred to [ESRGAN](https://github.com/xinntao/ESRGAN).

- **Relativistic LSGAN.**
- **Changed to use "Group Normalization" layers instead of "Batch Normalization" layers.**
- **Increased channels in Generator.**
- **Residual in Residual Dense Block (RRDB) structure in Generator.**
- **Changed to use "Swish" activation function instead of "ReLU".**
- **Using MAE (Mean Absolute Error) loss.**
- **Using small weight initialization.**
- **Without "VGG".**

### This model is incomplete for now.

### System Requirements
- **Memory: 12GB RAM**


### Preparation

We run this script under [TensorFlow](https://www.tensorflow.org) 1.12 and the [TensorLayer](https://github.com/tensorlayer/tensorlayer) 1.8.0+.

1. Install TensorFlow.

1. Follow the instructions below to install other requirements.
```bash
cd ~/
sudo python3 -m pip install https://github.com/tensorlayer/tensorlayer/archive/master.zip
sudo python3 -m pip install --upgrade tensorlayer
git clone https://github.com/ImpactCrater/SRGAN-R.git
sudo python3 -m pip install easydict
sudo apt install python3-tk
```


### Prepare Data

 - You need to have the high resolution images for training and validation.
   -  You can set the path to your training image folder via `config.TRAIN.hr_img_path` in `config.py`.
   -  You can set the path to your validation image folder via `config.VALID.hr_img_path` in `config.py`.
   -  Subdirectories are searched recursively.



### Run

#### Start training.

```bash
python main.py
```

#### Start evaluation.
 - After training, if you want to test the model, You need to put the image in the specified folder.
   -  You can set the path to your test image folder via `config.VALID.eval_img_path` in `config.py`.
   -  You can set the name of your test image via `config.VALID.eval_img_name` in `config.py`.(Default; "1.png")
  

```bash
python main.py --mode=evaluate 
```


## About original SRGAN

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)

### Author
- [zsdonghao](https://github.com/zsdonghao)

### Citation
If you find this project useful, we would be grateful if you cite the TensorLayer paper：

```
@article{tensorlayer2017,
author = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
journal = {ACM Multimedia},
title = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
url = {http://tensorlayer.org},
year = {2017}
}
```

### Other Projects

- [Style Transfer](https://github.com/tensorlayer/adaptive-style-transfer)
- [Pose Estimation](https://github.com/tensorlayer/openpose)

### Discussion

- [TensorLayer Slack](https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc)
- [TensorLayer WeChat](https://github.com/tensorlayer/tensorlayer-chinese/blob/master/docs/wechat_group.md)

### License

- For academic and non-commercial use only.
- For commercial use, please contact tensorlayer@gmail.com.
