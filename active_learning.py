import tensorflow as tf
import numpy as np
import sys
import os
import glob
import cv2
from detection import ObjectDetector
from config import DETECTION_MODEL, DETECTION_CATEGORIES
import time

DETECTION_THRESHOLD = 0.2


if __name__=="__main__":
    # check if video path is given as input
    if len(sys.argv) < 2:
        print("Input video argument missing...")
        exit()

    if len(sys.argv) == 2:
        video = sys.argv[1]

    # create export folder 
    export_folder_path = video.split('.')[0]

    # check if folder already exists and get user input
    if os.path.exists(export_folder_path):
        #print('folder already exists... Do you want to override existing files?[y]es,[n]o')
        while(1):
            user_input = input('folder already exists... Do you want to override existing files? Type [yes] or [no]')
            if user_input == 'no':
                exit()
            if user_input == 'yes':
                break
    else:
        os.mkdir(export_folder_path)

    # open video capture
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    num_frames = 0
    bad_frames = 0

    # load detection model 
    detection_model = ObjectDetector(DETECTION_MODEL, DETECTION_CATEGORIES)
    
    # predict boxes
    # if trigger
        # save image (image name = frame number)
        # save xml with predictions
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # process frame
            raw_img = frame.copy()
            timer = cv2.getTickCount()
            
            flag = False
            num_detections = 0
            num_good_detections = 0
            detection_model.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            for i in range(detection_model.detections['detection_boxes'].shape[0]):
                detection_score = detection_model.detections['detection_scores'][i] 
                if (detection_score > DETECTION_THRESHOLD) :
                    ymin, xmin, ymax, xmax = detection_model.detections['detection_boxes'][i]
                    detection_class = detection_model.detections['detection_classes'][i]
                    object_name = detection_model.category_index[detection_class]['name']

                    # show detection
                    if object_name == 'beet': 
                        num_detections +=1
                        if detection_score > 0.8:
                            num_good_detections +=1
                        cv2.rectangle(frame, (int(xmin*W), int(ymin*H)), (int(xmax*W), int(ymax*H)), (213, 255, 0), 5)
                        cv2.putText(frame, object_name, (int(xmin*W), int(ymin*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (213, 255, 0), 2, cv2.LINE_AA)
                    if object_name == 'weed':
                        cv2.rectangle(frame, (int(xmin*W), int(ymin*H)), (int(xmax*W), int(ymax*H)), (66, 66, 245), 5)
                        cv2.putText(frame, object_name, (int(xmin*W), int(ymin*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (66, 66, 245), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'{detection_score:.2f}', (int(xmin*W), int(ymax*H)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                    # check for triggers
                    cv2.putText(frame, str(int(fps)), (0, int(H)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            num_frames +=1



            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if (num_good_detections <= 1) and (num_detections >= 4) or (num_good_detections - num_detections) < -4:
                bad_frames +=1
                filename = os.path.join(export_folder_path, str(num_frames) + '.jpg')
                print('saving image: ', filename)
                cv2.imwrite(filename, raw_img)

                # create xml with detections
        else:
            break

    print('bad frames ', bad_frames, 'of total ', num_frames)
    
    cap.release()
    cv2.destroyAllWindows()



