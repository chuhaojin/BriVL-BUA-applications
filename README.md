# BriVL for Building Applications

This repo is used for illustrating how to build applications by using **BriVL** model

### This repo is re-implemented from following projects:

- [Source Code of BriVL 1.0](https://github.com/BAAI-WuDao/BriVL)
- [Model of BriVL 1.0](https://wudaoai.cn/model/detail/BriVL) 
- [Bottom Up Attention For Application](https://github.com/chuhaojin/bottom-up-attention-ForApp)
- [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
- [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch)

## Online Demo built by BriVL

- [AI's Imaginary World](http://buling.wudaoai.cn/)

- [Soul Music](https://weixin.elensdata.com/)

- [Few Word](http://120.92.50.21:6177/)


## Contents
This repo contains two parts:
- Bounding Box Extractor: ./bbox_extractor
- BriVL Feature Extractor: ./BriVL


## Test this Pipeline

Test image has been saved in **./bbox_extractor/feature_extractor**, test with following command:

```
python3 main.py --brivl_cfg BriVL/cfg/BriVL_cfg.yml --brivl_weights BriVL/weights/brivl-weights.pth
```



## Download Models

- [bua-caffe-frcn-r101\_with\_attributes.pth](https://drive.google.com/file/d/1oquCwDEvuJPeU7pyPg-Yudj5-8ZxtG0W/view) -> /bbox_extractor/weights
- [chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)  ->  /BriVL/weights/hfl
- [tf_efficientnet_b5_ns-6f26d0cf.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth) ->  /BriVL/weights
- [brivl-weights.pth\*](https://wudaoai.cn/model/detail/BriVL) ->  /BriVL/weights

## Requirements

- [Python](https://www.python.org/downloads/) >= 3.6
- [PyTorch](http://pytorch.org/) >= 1.4
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.2 and [cuDNN](https://developer.nvidia.com/cudnn)
- [Detectron2](https://github.com/facebookresearch/detectron2/releases/tag/v0.3) <= 0.3
- [Transformers](https://github.com/huggingface/transformers) 

**Important: The version of Detectron2 should be 0.3 or below.**


## More Resources

[Source Code of BriVL 1.0](https://github.com/BAAI-WuDao/BriVL)

[Model of BriVL 1.0\*](https://wudaoai.cn/model/detail/BriVL) 

[Online API of BriVL 1.0](https://github.com/chuhaojin/WenLan-api-document)

[Online API of BriVL 2.0](https://wudaoai.cn/model/detail/BriVL)

\* indicates an application is needed.

## Contact
This repo is maintained by Chuhao JIn([@jinchuhao](https://github.com/chuhaojin)).
