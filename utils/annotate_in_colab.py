import os 
import json 
import cv2
import random
import copy

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
    "category_id": 0
}


def scale_bbox(bboxes,image_paths):
    scaled_bboxes = bboxes
    return scaled_bboxes

def build_dataset(image_paths, training_data, boxes):
    os.system("mkdir annotations")
    im_id = 0
    
    for bbox,image,path in zip(boxes,training_data,image_paths):
        
        if bbox = None:
            continue
        #create datapoint
        datapoint = copy.deepcopy(datapoint_template)
        datapoint["file_name"] = path 
        datapoint["height"] = image.shape[0] 
        datapoint["width"] = image.shape[1] 
        datapoint["image_id"] = im_id 
        
        im_id+=1
        
        ann = copy.deepcopy(annotaion_template)
        ann["bbox"] = [bbox[0][0]*image.shape[1],bbox[0][1]*image.shape[0],
                       bbox[0][2]*image.shape[1],bbox[0][3]*image.shape[0]]
    
        datapoint["annotations"].append(ann)
        
        with open("annotations/" + path.split('/')[1][:-4] + "_annotation.json", "w") as outfile:
                json.dump(datapoint, outfile)

        
    print("output annotations folder:",outfile)
    return 

def setup_gui():
    
    os.system("pip install git+git://github.com/ricardodeazambuja/colab_utils.gits")
    
    print("big shoutout to ricardodeazambuja for making such a wonderful tool!")
    
    print("in colab GUI setup!")
    
    return 

def sample_frames(video_path,total_num_frames):
    output_path = 'images/'
    os.system('mkdir ' + output_path)
    
    if type(video_path) is not list:
        video_paths = [video_path]
    else:
        video_paths = [ video_path + x for x in os.listdir(video_path) if x[-3:] == "avi"]

    video_number = 0
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_num_frames):
            frame_no = random.randint(1, num_frames - 1)
            cap = cv2.VideoCapture(path)
            cap.set(1, frame_no)
            ret, frame = cap.read()
            cv2.imwrite(
                output_path
                + "video_"
                + str(video_number)
                + "_frame_"
                + str(frame_no)
                + ".jpg",
                frame,
            )
        cap.release()
        video_number += 1

    print("sampling frames randomly")
    print("output image folder:",output_path)
    return 

def sample_frames_feature_spacing(video_paths):
    print("sampling frames by finding a maximal spaning feature set")
    print("output image folder:")
    return 

