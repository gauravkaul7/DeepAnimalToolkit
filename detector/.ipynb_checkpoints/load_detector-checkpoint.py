class Detector:
    """
    this class trains a faster rcnn with selected branches(keypoint,mask,etc.)
    branch: either "mask", "keypoint", or "none"
    """

    def __init__(self,weights_path: "none"):
        from detectron2 import model_zoo
        
        cfg = get_cfg()

        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
        cfg.TEST.DETECTIONS_PER_IMAGE = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.WEIGHTS = weights_path

        self.detection_model = DefaultPredictor(cfg)

    def get_detections_video(video_path str:):
        cap = cv2.VideoCapture(video)
        for f in tqdm(range(int(total_frames-1))):
            ret, frame = cap.read()
            outputs = self.detection_model(frame)
            detections = outputs["instances"].to("cpu").pred_boxes.tensor.numpy() # np.array(detections)
            out_scores = outputs["instances"].to("cpu").scores.numpy() # np.array(out_scores) 
            conf_scores = []
            b_boxes = []
            deep_features = []
    
    def get_detections_image(image_path str:):
        cap = cv2.VideoCapture(video)
        for f in tqdm(range(int(total_frames-1))):
            ret, frame = cap.read()
            outputs = self.detection_model(frame)
            detections = outputs["instances"].to("cpu").pred_boxes.tensor.numpy() # np.array(detections)
            out_scores = outputs["instances"].to("cpu").scores.numpy() # np.array(out_scores) 
            conf_scores = []
            b_boxes = []
            deep_features = []