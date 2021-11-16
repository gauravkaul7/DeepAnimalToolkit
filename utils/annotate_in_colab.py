import os 
import json 
import cv2


def scale_bbox(bboxes,image_paths):
    
    scaled_bboxes = bboxes
    return scaled_bboxes

def build_annotations(scaled_bboxes,image_paths):
    print("output annotations folder:")
    return 

def setup_gui():
    os.system("git clone --depth 1 https://github.com/tensorflow/models")
    
    sh = '''
    
    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install -q .
    
    '''
    
    with open('script.sh', 'w') as file:
        file.write(sh)
    os.system("bash script.sh")
    
    print("in colab GUI setup!")
    
    return 0

def sample_frames(video_path):
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
        for i in range(args.num_samples):
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
    print("output image folder:")
    return 

def sample_frames_feature_spacing(video_paths):
    print("sampling frames by finding a maximal spaning feature set")
    print("output image folder:")
    return 

