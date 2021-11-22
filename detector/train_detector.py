import torch, torchvision
import numpy as np
import os, json, cv2, random

from google.colab.patches import cv2_imshow

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetMapper 
from detectron2.data import build_detection_train_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

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
                ann["category_id"] = 0
        
        print("filterig dataset  ...")
        print("Loaded " + str(len(dataset_dicts)) + " training datapoints")
        
        return dataset_dicts

    def load_dataset(self, annotations_folder: str):
        print("looking in", annotations_folder, "for annotations")
        DatasetCatalog.register(
            "train_dataset", lambda p=annotations_folder: self.get_dataset_dicts(p)
        )
        MetadataCatalog.get("train_dataset").set(thing_classes=["tracking target"])
        self.metadata = MetadataCatalog.get("train_dataset")
        self.visualize_examples(annotations_folder)

    def train_detector(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.cfg.DATASETS.TRAIN = ("train_dataset",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.MAX_ITER = 500
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
        self.trainer = DefaultTrainer(self.cfg) 
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        
    def visualize_examples(self,annotations_path):
        print('visualizing some examples')
        dataset_dicts = self.get_dataset_dicts(annotations_path)
        for d in random.sample(dataset_dicts, len(dataset_dicts)//5):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2_imshow(out.get_image()[:, :, ::-1])
        