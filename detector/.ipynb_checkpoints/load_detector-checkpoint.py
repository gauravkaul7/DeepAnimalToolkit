import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

import torch, torchvision

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

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
    this class trains a faster rcnn with selected branches(keypoint,mask,etc.)
    branch: either "mask", "keypoint", or "none"
    """

    def __init__(self, model_type: str):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(pretrained_weights[model_type])
        )
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        

    def load_trained_model(self, weights_path: str):
        self.cfg.MODEL.WEIGHTS = weights_path
        self.detection_model = DefaultPredictor(self.cfg)
        print("loaded model from:", weights_path)


    def get_detections_video(video_path):

        cap = cv2.VideoCapture(video_path)
        detections = []

        for f in tqdm(range(int(total_frames - 1))):
            ret, frame = cap.read()

            with torch.no_grad():
                outputs = self.detection_model(frame)

            if outputs["instances"].to("cpu").scores.shape == 0:
                detections.append([0, -1, -1, -1, -1])
            else:
                detections.append(
                    [1] + outputs["instances"].to("cpu").pred_boxes.tensor.tolist()
                )
        return detections
