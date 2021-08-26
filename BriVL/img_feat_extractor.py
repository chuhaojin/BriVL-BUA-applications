# -*- encoding: utf-8 -*-
'''
@File    :   text_feat_extractor.py
@Time    :   2021/08/26 10:46:15
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib

import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from utils import getLanMask
from utils.config import cfg_from_yaml_file, cfg
from models.vl_model import *

class ImgModel(nn.Module):
    def __init__(self, model_cfg):
        super(ImgModel, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['imgencoder'] = ImgLearnableEncoder(model_cfg)

    def forward(self, imgFea, maskImages, image_boxs):
        imgFea = self.learnable['imgencoder'](imgFea, maskImages, image_boxs) # <bsz, img_dim>
        imgFea = F.normalize(imgFea, p=2, dim=-1)
        return imgFea

class ImgFeatureExtractor:
    def __init__(self, cfg_file, model_weights, gpu_id = 0):
        self.gpu_id = gpu_id
        self.cfg_file = cfg_file
        self.cfg = cfg_from_yaml_file(self.cfg_file, cfg)
        self.img_model = ImgModel(model_cfg=self.cfg.MODEL)

        self.img_model = self.img_model.cuda(self.gpu_id)
        model_component = torch.load(model_weights, map_location=torch.device('cuda:{}'.format(self.gpu_id)))
        img_model_component = {}
        for key in model_component["learnable"].keys():
            if "imgencoder." in key:
                img_model_component[key] = model_component["learnable"][key]
        self.img_model.learnable.load_state_dict(img_model_component)
        self.img_model.eval()
        self.visual_transform = self.visual_transforms_box(self.cfg.MODEL.IMG_SIZE)

    def visual_transforms_box(self, new_size = 456):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((new_size, new_size)),
                normalize])

    def extract(self, img_path, bboxes):
        image = Image.open(img_path).convert('RGB')
        if image is None:
            return None
        else:
            width, height = image.size
            new_size = self.cfg.MODEL.IMG_SIZE
            img_box_s = []
            for box_i in bboxes[:self.cfg.MODEL.MAX_IMG_LEN-1]: # [x1, y1, x2, y2]
                x1, y1, x2, y2 = box_i[0] * (new_size/width), box_i[1] * (new_size/height), box_i[2] * (new_size/width), box_i[3] * (new_size/height)
                img_box_s.append(torch.from_numpy(np.array([x1, y1, x2, y2]).astype(np.float32)))     
            img_box_s.append(torch.from_numpy(np.array([0, 0, new_size, new_size]).astype(np.float32)))

            image_boxs = torch.stack(img_box_s, 0) # <36, 4>
            image = self.visual_transform(image)
            img_len = torch.full((1,), self.cfg.MODEL.MAX_IMG_LEN, dtype=torch.long)

            with torch.no_grad():
                imgs = image.unsqueeze(0)  # <batchsize, 3, image_size, image_size>
                img_lens = img_len.unsqueeze(0).view(-1)
                image_boxs = image_boxs.unsqueeze(0) # <BSZ, 36, 4>

                # get image mask
                imgMask = getLanMask(img_lens, cfg.MODEL.MAX_IMG_LEN)
                imgMask = imgMask.cuda(self.gpu_id)

                imgs = imgs.cuda(self.gpu_id)
                image_boxs = image_boxs.cuda(self.gpu_id) # <BSZ, 36, 4>
                img_fea = self.img_model(imgs, imgMask, image_boxs)
                img_fea = img_fea.cpu().numpy()
            return img_fea


if __name__ == '__main__':
    cfg_file = 'cfg/BriVL_cfg.yml'
    model_weights = "weights/BriVL-1.0-5500w.pth"    
    img_path = '../bbox_extractor/feature_extractor/test.png'
    bboxes = np.load('../bbox_extractor/feature_extractor/test.npz')['bbox'].tolist()
    vf_extractor = ImgFeatureExtractor(cfg_file, model_weights)
    print("img_path:", img_path)
    fea = vf_extractor.extract(img_path, bboxes)
    print("fea:", fea)
