import tensorflow as tf 
import numpy as np
from object_detection.utils import dataset_util
import sys
from pathlib import Path
import cv2

#raw_dataset = tf.data.TFRecordDataset("/home/lukas/training_workspace/data/beet/TFRecords/2021-05-14_07-20-57_camera_1.record")

if __name__=='__main__':
    if len(sys.argv) < 3:
        print('Enter [TFRecord_path] [image_folder_path]')
        sys.exit()

    if len(sys.argv) == 3:
        tf_record_path = sys.argv[1]
        image_folder_path = sys.argv[2]

    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    original_folder = image_folder_path + tf_record_path.split('/')[-1].split('.')[0]
    print(original_folder)






for raw_record in raw_dataset.take(10):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    info = {}
    for k, v in example.features.feature.items():
        #print(k)
        if k == 'image/filename':
            info[k] = v.bytes_list.value[0]
        if k == 'image/encoded':
            #print(v.bytes_list.value[0])
            info[k] = v.bytes_list.value[0]
        elif k in ['image/height', 'image/width']:
#            print(v.int64_list.value[0])
            info[k] = v.int64_list.value[0]
        elif k == 'image/key/sha256':
            info[k] = v.bytes_list.value[0]
        elif k == 'image/object/bbox/xmin':
            info[k] = (v.float_list)
        elif k == 'image/object/bbox/xmax':
            info[k] = (v.float_list)
        elif k == 'image/object/bbox/ymin':
            info[k] = (v.float_list)
        elif k == 'image/object/bbox/ymax':
            info[k] = (v.float_list)
        elif k == 'image/object/class/text':
            print(v.bytes_list.value)

        


    img = cv2.imread(original_folder + '/' + info['image/filename'].decode('utf-8'))
    
    for i, v in enumerate(info['image/object/bbox/xmin'].value):
        cv2.rectangle(img, (int(info['image/object/bbox/xmin'].value[i]*info['image/width']),
                            int(info['image/object/bbox/ymin'].value[i]*info['image/height'])),
                           (int(info['image/object/bbox/xmax'].value[i]*info['image/width']),
                            int(info['image/object/bbox/ymax'].value[i]*info['image/height'])),
                           (255,255,0),
                           1)
    
    

    cv2.imshow('img', img)
    cv2.waitKey(0)
    
    #img_arr = np.frombuffer(info['image/encoded'], dtype = np.uint8).reshape(info['image/height'], info['image/width'], 3)
    #cv2.imshow('img', img_arr)
    #cv2.waitKey(0)
