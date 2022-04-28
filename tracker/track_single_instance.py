import numpy as np
import csv

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
import filterpy

import cv2
from tqdm import tqdm
import argparse

pretrained_weights = {
    "mask_50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "mask_101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "keypoint_50": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    "keypoint_101": "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
    "detector_50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "detector_101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
}


class SingleInstanceTracker:
    """
    This class implements single object tracker. Since there is a known single object we do not need to associate object IDs,
    but we need to filter points. This is becasue we assume our detections are noisy estimates of our objects location
    and/or keypoints.
    """

    def __init__(self, model_type: str, weights_path: str):
        
        self.cfg = get_cfg()
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.merge_from_file(
            model_zoo.get_config_file(pretrained_weights[model_type])
        )
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.detector = DefaultPredictor(self.cfg)
        
        self.dt = 0.7

        # create sigma points to use in the filter. This is standard for Gaussian processes
        self.points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)

        self.kf = UnscentedKalmanFilter(
            dim_x=4, dim_z=2, dt=self.dt, fx=self.fx, hx=self.hx, points=self.points
        )

        self.kf.P = 0.95  # initial uncertainty
        self.z_std = 0.1
        self.kf.R = np.diag([self.z_std**2, self.z_std**2])  # 1 standard
        self.kf.Q = filterpy.common.Q_discrete_white_noise(
            dim=2, dt=self.dt, var=0.01**2, block_size=2
        )

    def get_detections_video(self, video_path, num_frames=-1):

        unfiltered_detections = []

        frame_array = []

        cap = cv2.VideoCapture(video_path)

        if num_frames == -1:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2

        for f in tqdm(range(num_frames)):

            ret, frame = cap.read()
            outputs = self.detector(frame)
            detections = (
                outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
            )  # np.array(detections)
            unfiltered_detections.append([f, detections])

        object_centerpoints = []

        for d in unfiltered_detections:
            if d[1].shape[0] > 0:
                x = (d[1][0][0] + d[1][0][2]) // 2
                y = (d[1][0][1] + d[1][0][3]) // 2
                object_centerpoints.append([x, y])
            else:
                if len(object_centerpoints) == 0:
                    object_centerpoints.append([0, 0])
                else:
                    object_centerpoints.append(object_centerpoints[-1])

        return object_centerpoints

    def track_object_offline(self, video_path, num_frames, output_path):

        detections = self.get_detections_video(video_path, num_frames)
        
        self.kf.x = np.array(
            [-1.0, detections[0][0], -1.0, detections[0][1]]
        )  # initial state
        trajectory_filtered = []
        for z in detections:
            self.kf.predict()
            self.kf.update(z)
            trajectory_filtered.append(self.hx(self.kf.x))

        trajectory_filtered = [
            [i, x[0], x[1]] for i, x in enumerate(trajectory_filtered)
        ]
        
        trajectory_filtered = [["frame id", "x", "y"]] + trajectory_filtered

        with open(output_path+".csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(trajectory_filtered)

        return trajectory_filtered

    def fx(self, x, dt):
        # state transition function - predict next state based
        # on constant velocity model x = vt + x_0
        F = np.array(
            [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=float
        )
        return np.dot(F, x)

    def hx(self, x):
        return np.array([x[0], x[2]])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Description of your program")
    
    parser.add_argument(
    "-v", "--video", help="video path", required=True
    )
    
    parser.add_argument(
    "-o", "--output_path", help="video path", required=True
    )
    
    parser.add_argument(
    "-m", "--model_path", help="video path", required=True
    ) 
    
    parser.add_argument(
    "-t", "--model_type", help="video path", required=True
    )
    
    args = vars(parser.parse_args())

    video_path = args["video"]
    output_path = args["output_path"]
    model_type = args["model_type"]
    weights_path = args["model_path"]
    
    tracker = SingleInstanceTracker(model_type, weights_path)
    tracker.track_object_offline(video_path, -1, output_path)

