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
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from transformers import AutoTokenizer

from utils import getLanMask
from utils.config import cfg_from_yaml_file, cfg
from models.vl_model import *

class TextModel(nn.Module):
    def __init__(self, model_cfg):
        super(TextModel, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['textencoder'] = TextLearnableEncoder(model_cfg)

    def forward(self, texts, maskTexts):
        textFea = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
        textFea = F.normalize(textFea, p=2, dim=-1)
        return textFea

class TextFeatureExtractor:
    def __init__(self, cfg_file, model_weights, gpu_id = 0):
        self.gpu_id = gpu_id
        self.cfg_file = cfg_file
        self.cfg = cfg_from_yaml_file(self.cfg_file, cfg)
        self.cfg.MODEL.ENCODER = os.path.join(base_dir, self.cfg.MODEL.ENCODER)
        self.text_model = TextModel(model_cfg=self.cfg.MODEL)

        self.text_model = self.text_model.cuda(self.gpu_id)
        model_component = torch.load(model_weights, map_location=torch.device('cuda:{}'.format(self.gpu_id)))
        text_model_component = {}
        for key in model_component["learnable"].keys():
            if "textencoder." in key:
                text_model_component[key] = model_component["learnable"][key]
        self.text_model.learnable.load_state_dict(text_model_component)
        self.text_model.eval()
        
        self.text_transform = AutoTokenizer.from_pretrained(self.cfg.MODEL.ENCODER)

    def extract(self, text_input):
        if text_input is None:
            return None
        else:
            text_info = self.text_transform(text_input, padding='max_length', truncation=True,
                                            max_length=self.cfg.MODEL.MAX_TEXT_LEN, return_tensors='pt')
            text = text_info.input_ids.reshape(-1)
            text_len = torch.sum(text_info.attention_mask)
            with torch.no_grad():
                texts = text.unsqueeze(0) 
                text_lens = text_len.unsqueeze(0)
                textMask = getLanMask(text_lens, cfg.MODEL.MAX_TEXT_LEN)
                textMask = textMask.cuda(self.gpu_id)
                texts = texts.cuda(self.gpu_id)
                text_lens = text_lens.cuda(self.gpu_id)
                text_fea = self.text_model(texts, textMask)
                text_fea = text_fea.cpu().numpy()
            return text_fea


if __name__ == '__main__':
    cfg_file = os.path.join(base_dir, 'cfg/BriVL_cfg.yml')
    model_weights = os.path.join(base_dir, "weights/BriVL-1.0-5500w.pth")
    vfe = TextFeatureExtractor(cfg_file, model_weights)
    text_query = "北京的秋天是真的凉爽。"
    print("text query:", text_query)
    fea = vfe.extract(text_query)
    print("fea:", fea)
