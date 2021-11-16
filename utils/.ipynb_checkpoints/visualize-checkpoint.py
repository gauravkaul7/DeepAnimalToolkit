import cv2

def visualizer():
    frame_array = []
    cap = cv2.VideoCapture("/content/single_mouse_test_video.avi")
    #cap.set(1,840464//2.12)
    cap.set(1,840464//1.7)
    num_frames = 200
    location = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    visualize = True

    for f in tqdm(range(num_frames)):
        ret, frame = cap.read()
        outputs = predictor(frame)
        detections = outputs["instances"].to("cpu").pred_boxes.tensor.numpy() # np.array(detections)
        center =  (int(detections[0][0] + (detections[0][2]-detections[0][0])//2) 
                ,int(detections[0][1] + (detections[0][3]-detections[0][1])//2)) 
        if visualize:
        frame = cv2.circle(frame, center, 4, (255,255,0), 3)
        height, width, layers = frame.shape

        new_h = int(height)
        new_w = int(width)
        frame = cv2.resize(frame, (new_w, new_h))
        size = (new_w,new_h)
        frame_array.append(frame)
        location.append(center)

        if visualize: 
            videoOut = "/content/out.mp4"
            VideoWriter = cv2.VideoWriter(videoOut,cv2.VideoWriter_fourcc('M','J','P','G'),15,size)

        for frame in frame_array:
            VideoWriter.write(frame)
        VideoWriter.release()

        # Input video path
        save_path = videoOut

        # Compressed video path
        compressed_path = videoOut[:-4]+"_comp.mp4"

        os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

        # Show video
        mp4 = open(compressed_path,'rb').read()

        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        HTML("""<video width=400 controls>
                    <source src="%s" type="video/mp4">
              </video>
          """ % data_url)
        return 0 