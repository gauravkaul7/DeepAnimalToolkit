import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

import torch, torchvision

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from tqdm import tqdm

pretrained_weights = {
    "mask_50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "mask_101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "keypoint_50": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    "keypoint_101": "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
    "detector_50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "detector_101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
}


class Detector:
    """
    this class loads the weights of a faster rcnn with selected branches(keypoint,mask,etc.)
    branch: either "mask", "keypoint", or "none"
    """

    def __init__(self, model_type: str, weights_path: str):
        self.cfg = get_cfg()
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
        self.cfg.merge_from_file(model_zoo.get_config_file(pretrained_weights[model_type]))
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.predictor = DefaultPredictor(self.cfg)
        print("loaded model from:", weights_path)

    def get_detections_video(self,video_path,num_frames):
        
        unfiltered_detections = []
        
        frame_array = []
        
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #cap.set(2,1889)
        
        print('total_frames:',total_frames)
        
        for f in tqdm(range(num_frames)):
            
            ret, frame = cap.read()
            outputs =  self.predictor(frame)
            detections = outputs["instances"].to("cpu").pred_boxes.tensor.numpy() # np.array(detections)
            out_scores = outputs["instances"].to("cpu").scores.numpy() # np.array(out_scores) 
            unfiltered_detections.append([f,detections])
            
        return unfiltered_detections

    def get_detector(self):
        return self.detection_model
