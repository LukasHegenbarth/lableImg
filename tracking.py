import numpy as np 
import cv2
from skimage.measure import ransac
from helpers import add_ones, poseRt, fundamentalToRt, normalize, EssentialMatrixTransform, myjet

img1 = cv2.imread('/home/lukas/Videos/2021-05-14_08-08-09_camera_1/125.jpg')
img2 = cv2.imread('/home/lukas/Videos/2021-05-14_08-08-09_camera_1/129.jpg')

img1 = cv2.resize(img1, (img1.shape[1]//4, img1.shape[0]//4))
img2 = cv2.resize(img2, (img2.shape[1]//4, img2.shape[0]//4))

video = cv2.VideoCapture('/home/lukas/Videos/2021-05-14_07-39-16_camera_1.mkv')

tracker_mosse = cv2.TrackerMOSSE_create
tracker_crt = cv2.TrackerCSRT_create

class MultiTracker:
  def __init__(self):
    self.trackers = []
    self.success = []
    self.boxes = []

  def add(self,tracker_type, frame, bbox):
    if tracker_type == 'BOOSTING':
      tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
      tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
      tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
      tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
      tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
      tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
      tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
      tracker = cv2.TrackerCSRT_create()
    else: 
      print('Unknown tracker type')
      return

    tracker.init(frame, bbox)
    self.trackers.append(tracker)

  def remove(self, idx):
    self.trackers.pop(idx)
    self.success.pop(idx)
    self.bbox.pop(idx)

  def update(self, frame):
    for idx, tracker in enumerate(self.trackers):
      success, bbox = tracker.update(frame)
      self.success[idx] = success
      self.bbox[idx] = bbox
      
    return (self.success, self.bbox)










multi_tracker = MultiTracker()


# create tracker in list for each detection bbox
# if tracker already exists, skip
multi_tracker.add('MOSSE', frame, bbox)
# update all trackers
(success, boxes) = multi_tracker.update(frame)

for box in boxes:
  (x,y,w,h) = [int(v) for v in box]
  cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)



 
# Exit if video not opened.
if not video.isOpened():
  print ("Could not open video")
  sys.exit()
 
# Read first frame.
for _ in range(30):
  video.read()
  
ok, frame = video.read()
frame = cv2.resize(frame, (frame.shape[1]//2,frame.shape[0]//2))
if not ok:
  print ('Cannot read video file')
  sys.exit()

cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
