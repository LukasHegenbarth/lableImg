"""
This module detects objects in images

takes RGB image as input
"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)
# import pathlib
import tensorflow as tf
# import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
# import glob
# from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# matplotlib.get_backend() 
import warnings
import cv2

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')   # Suppress TensorFlow logging (2)



class ObjectDetector():
    def __init__(self, detection_model, detection_categories):
        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # load network
        print("Loading object detection network...")
        self.detect_fn = tf.saved_model.load(detection_model)
        print("Loading detection categories...")
        self.category_index = label_map_util.create_category_index_from_labelmap(detection_categories, use_display_name=True)
        
        

    # predict objects in image
    def detect(self, image_np):
        # return pixel coordinates
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        self.detections = self.detect_fn(input_tensor)
        num_detections = int(self.detections.pop('num_detections'))

        self.detections = {key: value[0, :num_detections].numpy()
            for key, value in self.detections.items()}

        self.detections['num_detections'] = num_detections

        self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)

    def calc_bbox_center(self, threshold):
        self.detections_center = []
        for i in range(self.detections['detection_boxes'].shape[0]):
            if self.detections['detection_scores'][i] > threshold:
                ymin, xmin, ymax, xmax = self.detections['detection_boxes'][i]
                x_center = xmin + ((xmax - xmin)/2)
                y_center = ymin + ((ymax - ymin)/2)
                self.detections_center.append((x_center, y_center))
            else:
                break

    def draw_bbox_center(self, img):
        height, width = img.shape[:2]
        centers = self.detections_center
        for center_coords in centers:
            x = int(center_coords[0] * width)
            y = int(center_coords[1] * height)
            cv2.circle(img, (x,y), 2, (0, 0, 255), 5)
        

    # draw bounding boxes of detections on input image
    # def draw_bounding_boxes(self, image, threshold):
    #     height, width = image.shape[:2]
    #     print("height, width:  ", height, width)
    #     for idx in range(self.detections['detection_boxes'].shape[0]):
    #         if self.detections['detection_scores'][idx] > threshold:
    #             print()
    #             ymin, xmin, ymax, xmax = self.detections['detection_boxes'][idx]
    #             (left, right, top, bottom) = (xmin * width, xmax * width,
    #                                         ymin * height, ymax * height)
    #             start_point = (int(left), int(top))
    #             end_point = (int(right), int(bottom))
    #             cv2.rectangle(image, start_point, end_point, color=(182, 66, 245) , thickness=2)
    #             plant_idx = self.detections['detection_classes'][idx]
    #             plant_name = self.category_index[plant_idx]['name']
    #             text = plant_name + ' ' + str(self.detections['detection_scores'][idx])
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             cv2.putText(image, text, start_point, font, 0.5, (0, 255, 0), 2)
    #         else:
    #             break

    def draw_bounding_boxes(self, image, threshold):
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image,
                self.detections['detection_boxes'],
                self.detections['detection_classes'],
                self.detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=threshold,
                agnostic_mode=False)





