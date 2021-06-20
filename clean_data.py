import sys
import xml.etree.ElementTree as ET
import numpy as np
from  scipy.spatial.distance import cdist
import glob
import time

#xml = '/home/lukas/Videos/2021-05-14_07-39-16_camera_1/29.xml'

def get_data_from_xml(xml):
  tree = ET.parse(xml)
  root = tree.getroot()

  num_objects = (len(root.findall('object')))
  bbox_array = np.empty([num_objects,4], dtype=int)
  object_type_list = []

  for idx,  annotation_object in enumerate(root.findall('object')):
    object_type = annotation_object.find('name').text
    object_type_list.append(object_type)
    bbox = annotation_object.find('bndbox')
    xmin = bbox.find('xmin').text
    ymin = bbox.find('ymin').text
    xmax = bbox.find('xmax').text
    ymax = bbox.find('ymax').text
    bbox_array[idx][:] = [xmin, ymin, xmax, ymax]
  
  return object_type_list, bbox_array

def calc_min_distance(points_2d):
  distance_mat = cdist(points_2d, points_2d)
  #print(distance_mat)
  min_distance = np.Inf
  row_idx, col_idx = np.triu_indices(distance_mat.shape[0],1)
  for i in range(len(row_idx)):
    if distance_mat[row_idx[i], col_idx[i]] < min_distance:
      min_distance = distance_mat[row_idx[i], col_idx[i]]

  return min_distance

  

if __name__=="__main__":
  if len(sys.argv) < 2:
    print("%s <folder with annotation files>" % sys.argv[0])
    exit(-1)
  
  start_time = time.time()
  xml_files = glob.glob(sys.argv[1] + '/*.xml')
  for xml in xml_files:
    object_types, bboxes = get_data_from_xml(xml)
    left_upper_corner = bboxes[:,0:1]
    right_upper_corner = bboxes[:,[2,1]]
    left_lower_corner = bboxes[:,[0,3]]
    right_lower_corner = bboxes[:,2:3]
    
    min_dist = calc_min_distance(left_upper_corner)

    if min_dist < 10:
      print(xml)
      print(min_dist)

  print('number of xml files parsed: ', len(xml_files))
  print('time elapsed: ', time.time()-start_time)
