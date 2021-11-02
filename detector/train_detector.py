import os
import json
import torch, torchvision

import detectron2

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper  # the default mapper
from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode

pretrained_weights = {
    "mask_50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "mask_101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "keypoint_50": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    "keypoint_101": "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
    "detector_50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "detector_101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
}


class DetectorTrainer:
    def __init__(self, model_type: str):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(pretrained_weights[model_type])
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            pretrained_weights[model_type]
        )
        print("starting model weights coming from:", pretrained_weights[model_type])

    def get_dataset_dicts(self, annotations_path: str):
        dataset_dicts = [
            json.load(open(annotations_path + x))
            for x in os.listdir(annotations_path)
            if x[-5:] == ".json"
        ]
        for datapoint in dataset_dicts:
            datapoint["file_name"] = datapoint["file_name"]
            for ann in datapoint["annotations"]:
                ann["bbox_mode"] = BoxMode.XYXY_ABS
                ann["segmentation"] = [ann["segmentation"]]
                ann["category_id"] = 0
        print("Loaded " + str(len(dataset_dicts)) + " training datapoints")
        print("filterig dataset ...")
        self.dataset_dicts = [
            x for x in dataset_dicts if len(x["annotations"][0]["segmentation"][0]) >= 6
        ]
        print("Loaded " + str(len(dataset_dicts)) + " training datapoints")
        return dataset_dicts

    def load_dataset(self, annotations_folder: str):
        print("looking in", annotations_folder, "for annotations")
        DatasetCatalog.register(
            "train_dataset", lambda p=annotations_folder: self.get_dataset_dicts(p)
        )
        MetadataCatalog.get("train_dataset").set(thing_classes=["mouse"])
        mouse_metadata = MetadataCatalog.get("train_dataset")
        self.cfg.DATASETS.TRAIN = ("train_dataset",)

    def train_detector(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.DATASETS.TRAIN = ("train_dataset",)
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.MAX_ITER = 100
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
