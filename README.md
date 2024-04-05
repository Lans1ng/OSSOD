# Semi-Supervised Object Detection with Uncurated Unlabeled Data for Remote Sensing Images

## 🚀 Introduction

<div align="center">
  <img width="300" src="resources/introduction.png"/>
</div>

Semi-supervised object detection (SSOD) assumes that both labeled and unlabeled data originate from the same label space, constituting in-distribution (ID) samples. Open-set semi-supervised object detection (OSSOD) accommodates the existence of substantial out-of-distribution (OOD) samples, mirroring the complexities of real-world scenarios. 


## 📻 Overview

<div align="center">
  <img width="450" src="resources/ossod.png"/>
</div>

Overview of the proposed open-set semi-supervised object detection framework.


## 🎮 Getting Started

### 1. Install Environment

see [INSTALL](INSTALL.md).

### 2. Prepare Dataset 

- DIOR Dataset ([Google Drive](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) or [BaiduNetDisk](https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg#list/path=%2F))

  ```shell
  dataset/
  	└── DIOR/
          ├── annotations_json_split1.json
          ├── annotations_json_split2.json
          └── JPEGImages
                ├── 00001.jpg
                ├── 00002.jpg
                ├── ...
                └── 23463.jpg
  ```

### 3. Download Checkpoints

Before training，please download the pretrained backbone ([ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth)) to pretrained_model/backbone.



### 4. Training

#### Use labeled data to train a baseline




## 🎫 License

The content of this project itself is licensed under [LICENSE](LICENSE).

## 💡 Acknowledgement

- [SSOD](https://github.com/hikvision-research/SSOD)
- [OWOD](https://github.com/JosephKJ/OWOD)


## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{liu2023semi,
  title={Semi-Supervised Object Detection with Uncurated Unlabeled Data for Remote Sensing Images},
  author={Liu, Nanqing and Xu, Xun and Gao, Yingjie and Li, Heng-Chao},
  journal={arXiv preprint arXiv:2310.05498},
  year={2023}
}
```