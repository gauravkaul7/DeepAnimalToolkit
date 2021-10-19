import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

import torch, torchvision

print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.8")

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class Detector:
    """
    this class trains a faster rcnn with selected branches(keypoint,mask,etc.)
    branch: either "mask", "keypoint", or "none"
    """

    def __init__(self, weights_path: "none"):
        from detectron2 import model_zoo

        cfg = get_cfg()
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
        cfg.TEST.DETECTIONS_PER_IMAGE = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
            )
        )

        cfg.MODEL.WEIGHTS = weights_path
        self.detection_model = DefaultPredictor(cfg)
        self.detection_model.eval()

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
