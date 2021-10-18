import os
import json

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.8")   # please manually install torch 1.8 if Colab changes its default version


import detectron2

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode



pretrained_weights = {'mask_50':'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
                      'mask_101':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
                      'keypoint_50':'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
                      'keypoint_101':'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
                      'detector_50':'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                      'detector_101':'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',}




class KeypointTrainer(DefaultTrainer): 
    @classmethod
    def build_train_loader(cls, cfg):
        mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((800, 800))])
        return build_detection_train_loader(cfg, mapper=mapper)
    

class DetectorTrainer:
    """
    this class trains a faster rcnn with selected branches(keypoint,mask,etc.)
    branch: either "mask", "keypoint", or "none"
    """
    def __init__(self, branch: "none"):
        
        self.cfg = get_cfg()
        
         
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))

        self.cfg.DATASETS.TRAIN = ("mouse_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")  
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  
        
        trainer = DefaultTrainer(cfg) 
        trainer = KeypointTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()
        
    def load_dataset(path):
        data_path = '/content/drive/MyDrive/keypoint_dataset/datasets/'
        annotations_folder = 'tracking_train_annotations/'
        def get_mouse_dicts(annotations_path):
            data_path = '/content/drive/MyDrive/keypoint_dataset/datasets/'
            dataset_dicts = [json.load(open(data_path+annotations_path+x)) for x in os.listdir(data_path+annotations_path) if x[-5:]=='.json']
            #dataset_dicts = [d for d in dataset_dicts if len([a for a in d["annotations"] if sum(a["keypoints"][2::3])!=16]) == 0]
            for datapoint in dataset_dicts:
                datapoint['file_name'] = data_path+datapoint['file_name'] 
                for ann in datapoint["annotations"]:
                    ann["bbox_mode"] = BoxMode.XYXY_ABS
                    ann["segmentation"] = [ann["segmentation"]]
                    ann["category_id"] = 0
            print("Loaded "+str(len(dataset_dicts))+" Examples")
            return dataset_dicts

        DatasetCatalog.register("mouse_train",lambda p=annotations_folder:get_mouse_dicts(p))
        MetadataCatalog.get("mouse_train").set(thing_classes=["mouse"])
        MetadataCatalog.get("mouse_train").set(keypoint_names=["s","fr","tb","fl"])
        MetadataCatalog.get("mouse_train").set(keypoint_flip_map=[])

        MetadataCatalog.get("mouse_train").set(keypoint_connection_rules=[("s","fr",(0,0,250)),("s","fl",(0,0,250)),
                                                                            ("fr","tb",(0,0,250)),("fl","tb",(0,0,250)),
                                                                            ("fr","fl",(0,0,250))])


        mouse_metadata = MetadataCatalog.get("mouse_train")
        cfg.DATASETS.TRAIN = ("mouse_train",)

    def train_detector(batch_size, iterations)
    
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = 1e-1 
        self.cfg.SOLVER.MAX_ITER = 1000
        #self.cfg.SOLVER.STEPS = []       
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4
        trainer = DefaultTrainer(cfg) 
        #trainer = KeypointTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()


    