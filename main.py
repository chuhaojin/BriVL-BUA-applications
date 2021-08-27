# -*- encoding: utf-8 -*-
'''
@File    :   BriVL_pipline.py
@Time    :   2021/08/26 12:49:25
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import os
import numpy as np

from BriVL.text_feat_extractor import TextFeatureExtractor
from BriVL.img_feat_extractor import ImgFeatureExtractor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--brivl_cfg', type=str, dest="brivl_cfg_file", default="BriVL/cfg/BriVL_cfg.yml")
parser.add_argument('--brivl_weights', type=str, dest="brivl_model_weights", default="BriVL/weights/BriVL-1.0-5500w.pth")
args = parser.parse_args()


# Text Feature Extractor
text_extra = TextFeatureExtractor(args.brivl_cfg_file, args.brivl_model_weights)

# Image Feature Extractor
img_extra = ImgFeatureExtractor(args.brivl_cfg_file, args.brivl_model_weights)

# Extract text feature.
text_query = "北京的秋天是真的凉爽。"

text_fea = text_extra.extract(text_query)
print("Text query:", text_query)
print("Text feature:", text_fea)

# Extract image feature.
img_path = 'bbox_extractor/test_data/test.png'
bboxes = np.load('bbox_extractor/test_data/test.npz')['bbox'].tolist()
img_fea = img_extra.extract(img_path, bboxes)
print("Image path:", img_path)
print("Image feature:", img_fea)

