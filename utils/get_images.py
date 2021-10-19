import cv2
import numpy as np
import os
import json
import random
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--input_video",
    help="input path to a folder of images you would like to create annotations for",
)
parser.add_argument(
    "--input_video_folder",
    help="input path to a folder of images you would like to create annotations for",
)
parser.add_argument(
    "--output_path",
    help="outputh path to a folder where you would like to store annotations",
)
parser.add_argument(
    "--num_samples",
    type=int,
    help="outputh path to a folder where you would like to store annotations",
)

args = parser.parse_args()

if args.input_video_folder == None:
    video_paths = [args.input_video]
else:
    video_paths = [
        args.input_video_folder + x
        for x in os.listdir(args.input_video_folder)
        if x[-3:] == "avi"
    ]


video_number = 0
for path in video_paths:
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(args.num_samples):
        frame_no = random.randint(1, num_frames - 1)
        cap = cv2.VideoCapture(path)
        cap.set(1, frame_no)
        ret, frame = cap.read()
        cv2.imwrite(
            args.output_path
            + "video_"
            + str(video_number)
            + "_frame_"
            + str(frame_no)
            + ".jpg",
            frame,
        )
    cap.release()
    video_number += 1

"""
import torch 
import torchvision.models as models
im = torch.rand(10,3,224,224)
resnet18 = models.resnet18(pretrained=True)
out = resnet18(im)
out_lengths = (torch.sum((out**2),dim=1)**.5)
unit_vectors = torch.divide(out,out_lengths.unsqueeze(1))
out = torch.mm(unit_vectors,unit_vectors.t())
print(out)"""
