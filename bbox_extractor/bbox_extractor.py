# -*- coding:UTF8 -*-

"""Image bounding-box extraction process."""

import os
import sys

import cv2
import numpy as np

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch

from utils.extract_utils import get_image_blob
from models import add_config
from models.bua.box_regression import BUABoxes
from models.bua import add_bottom_up_attention_config
from detectron2.layers.nms import nms

class BboxExtractor:
    def __init__(self, cfg_file, gpu_id = 0):
        
        self.cfg_file = cfg_file
        self.gpu_id = gpu_id
        self.cfg = get_cfg()
        add_bottom_up_attention_config(self.cfg, True)
        self.cfg.merge_from_file(self.cfg_file)
        self.cfg.freeze()
        default_setup(self.cfg, None)

        self.bbox_extract_model = DefaultTrainer.build_model(self.cfg)
#        self.bbox_extract_model.cuda(gpu_id)
        bbox_extract_model_dict = self.bbox_extract_model.state_dict()
        bbox_extract_checkpoint_dict = torch.load(self.cfg.MODEL.WEIGHTS, map_location=torch.device('cuda:0'))['model']
        bbox_extract_checkpoint_dict = {k:v for k, v in bbox_extract_checkpoint_dict.items() if k in bbox_extract_model_dict}
        bbox_extract_model_dict.update(bbox_extract_checkpoint_dict)
        self.bbox_extract_model.load_state_dict(bbox_extract_model_dict)
        # self.bbox_extract_model = torch.nn.DataParallel(self.bbox_extract_model, device_ids=self.gpus)
        self.bbox_extract_model.eval()

    def clean_bbox(self, dataset_dict, boxes, scores):
        MIN_BOXES = self.cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
        MAX_BOXES = self.cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
        CONF_THRESH = self.cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

        scores = scores[0]
        boxes = boxes[0]
        num_classes = scores.shape[1]
        boxes = BUABoxes(boxes.reshape(-1, 4))
        boxes.clip((dataset_dict['image'].shape[1]/dataset_dict['im_scale'], dataset_dict['image'].shape[2]/dataset_dict['im_scale']))
        boxes = boxes.tensor.view(-1, num_classes*4)  # R x C x 4

        cls_boxes = torch.zeros((boxes.shape[0], 4))
        for idx in range(boxes.shape[0]):
            cls_idx = torch.argmax(scores[idx, 1:]) + 1
            cls_boxes[idx, :] = boxes[idx, cls_idx * 4:(cls_idx + 1) * 4]

        max_conf = torch.zeros((scores.shape[0])).to(scores.device)
        for cls_ind in range(1, num_classes):
                cls_scores = scores[:, cls_ind]
                keep = nms(cls_boxes, cls_scores, 0.3)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                                 cls_scores[keep],
                                                 max_conf[keep])
                
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
        image_bboxes = cls_boxes[keep_boxes]

        return image_bboxes

    def extract_bboxes(self, img_path):
        im = cv2.imread(img_path)
        if im is None:
            print("img is None!")

            return 501, None
        else:
            dataset_dict = get_image_blob(im, self.cfg.MODEL.PIXEL_MEAN)
#            dataset_dict["image"] = dataset_dict["image"].cuda(self.gpu_id)
#            print(dataset_dict["image"])
#            print("-----model-----")
#            print(self.bbox_extract_model)
            with torch.set_grad_enabled(False):
                boxes, scores = self.bbox_extract_model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            boxes = self.clean_bbox(dataset_dict, boxes, scores)

            return 200, boxes # boxes type tensor

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, dest="img_path", default="test_data/test.png")
    parser.add_argument('--out_path', type=str, dest="output_file", default="test_data/test.npz")
    args = parser.parse_args()
    bbx_extr = BboxExtractor('configs/bua-caffe/extract-bua-caffe-r101.yaml')
    code, bboxes = bbx_extr.extract_bboxes(args.img_path)
    np.savez_compressed(args.output_file, bbox=bboxes)
    print(code)
    np_bbox = bboxes.numpy().astype(np.int32)
    print(np_bbox)
    print(bboxes.shape)

