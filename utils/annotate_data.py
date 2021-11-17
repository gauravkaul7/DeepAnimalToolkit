import cv2
import os
import argparse
import random
import copy
import json


parser = argparse.ArgumentParser(
    description="This script is used for annotating data and saving them as coco style annotations"
)

parser.add_argument(
    "--input_path",
    help="input path to a folder of images you would like to create annotations for",
)
parser.add_argument(
    "--output_path",
    help="outputh path to a folder where you would like to store annotations",
)
args = parser.parse_args()

images = [
    args.input_path + im
    for im in os.listdir(args.input_path)
    if im[:-4] + "_annotation.json" not in os.listdir(args.output_path)
    and im[-3:] == "jpg"
]


datapoint_template = {
    "file_name": "",
    "height": 0,
    "width": 0,
    "image_id": 0,
    "annotations": [],
}
annotaion_template = {
    "bbox": [],
    "bbox_mode": "BoxMode.XYXY_ABS",
    "category_id": 0,
    "segmentation": [],
    "keypoints": [],
}

# global variables

ix = -1
iy = -1

# color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
# instances = []
# pts = []


class annotationMode:
    def __init__(self):
        self.modes = [
            "segmentation",
            "keypoint - (visible+marked)",
            "keypoint - (not visible+marked)",
            "keypoint - (not visible+unmarked)",
        ]
        self.mode = self.modes[0]

    def get_mode(self):
        self.mode = self.modes[0]
        self.modes = self.modes[1:] + [self.modes[0]]
        return self.mode


def make_masks(event, x, y, flags, param):
    global ix, iy, pts, key_pts, color, instances
    ## this block will a polygon for a single instance
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y

    elif event == cv2.EVENT_LBUTTONUP:

        if mode_name == "segmentation":
            cv2.circle(img, (ix, iy), 1, color, -1)

            if len(pts) > 0:
                cv2.line(img, tuple(pts[-2:]), (ix, iy), color, 2)
            pts += [ix, iy]

        if mode_name == "keypoint - (visible+marked)":
            cv2.circle(img, (ix, iy), 2, color, -1)
            key_pts += [ix, iy, 2]

        if mode_name == "keypoint - (not visible+marked)":
            cv2.circle(img, (ix, iy), 2, color, -1)
            key_pts += [ix, iy, 1]

        if mode_name == "keypoint - (not visible+unmarked)":
            cv2.circle(img, (ix, iy), 3, color, -1)
            key_pts += [0, 0, 0]

    elif event == cv2.EVENT_RBUTTONDOWN:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        instances.append({"pts": pts, "key_pts": key_pts})
        key_pts = []
        pts = []


for img_path in images:
    mode = annotationMode()
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    instances = []
    pts = []
    key_pts = []
    img = cv2.imread(img_path)
    window_name = img_path.split("/")[-1]

    cv2.namedWindow(winname=window_name)
    cv2.setMouseCallback(window_name, make_masks)

    datapoint = copy.deepcopy(datapoint_template)
    datapoint_template = {
        "file_name": "",
        "height": 0,
        "width": 0,
        "image_id": 0,
        "annotations": [],
    }

    datapoint["file_name"] = img_path
    datapoint["height"] = img.shape[0]
    datapoint["width"] = img.shape[1]
    datapoint["image_id"] = len(os.listdir(args.output_path))

    mode_name = mode.get_mode()
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, mode_name, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    while True:
        cv2.imshow(window_name, img)

        if cv2.waitKey(10) == 32:
            mode_name = mode.get_mode()
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                img, mode_name, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        if cv2.waitKey(10) == 27:
            if len(pts) > 5:
                instances.append({"pts": pts, "key_pts": key_pts})

            for ann in instances:
                seg = ann["pts"]
                key_pts = ann["key_pts"]

                annotation = copy.deepcopy(annotaion_template)

                annotation["bbox"] = [
                    min([x for x in seg[::2]]),
                    min([y for y in seg[1::2]]),
                    max([x for x in seg[::2]]),
                    max([y for y in seg[1::2]]),
                ]

                annotation["segmentation"] = seg

                annotation["keypoints"] = key_pts

                annotation["num_keypoints"] = int(len(key_pts) / 3)

                print("num_keypoints:", annotation["num_keypoints"])

                datapoint["annotations"].append(annotation)

            with open(args.output_path + window_name.split(".")[0] + "_annotation.json", "w") as outfile:
                json.dump(datapoint, outfile)
                print(args.output_path + window_name.split(".")[0] + "_annotation.json")
            break

    cv2.destroyAllWindows()
