import cv2
from tqdm import tqdm

def visualize_tracking(video_path, trajectory, num_frames=0,scale=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames == 0: 
        num_frames = total_frames-2

    print('visualizing', num_frames,'frames') 

    frame_id = 0
    frame_array = []

    for f in tqdm(range(num_frames)):
        
        ret, frame = cap.read()
        frame = cv2.circle(frame, (int(trajectory[f][0]),int(trajectory[f][1])), 10, (255,0,0), 1)
        
        if f >= 6:
            trajectory_slice = [(int(x[0]),int(x[1])) for x in trajectory[f-4:f+1]]
            for c1,c2 in zip(trajectory_slice[::2],trajectory_slice[1::2]):
                frame = cv2.line(frame, c1, c2, (255,0,0), 4)
                
        height, width, layers = frame.shape
        new_h = int(height*scale)
        new_w = int(width*scale)
        frame = cv2.resize(frame, (new_w, new_h))
        size = (new_w,new_h)
        frame_array.append(frame)


    output_video = "tracking_visualized.mp4"
    VideoWriter = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc('M','J','P','G'),30,size)

    for frame in frame_array:
        VideoWriter.write(frame)
    
    VideoWriter.release()

