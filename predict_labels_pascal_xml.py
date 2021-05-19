import os
import sys

import cv2
import numpy as np
from xml.etree import ElementTree
import glob


from config import DETECTION_CATEGORIES, DETECTION_MODEL
from detection import ObjectDetector
# from display import Display2D

def prettify(element, indent='        '):
    queue = [(0, element)] # level, element
    while queue:
        level, element = queue.pop(0)
        children = [(level + 1, child) for child in list(element)]
        if children:
            element.text = '\n' + indent * (level+1) # for child open
        if queue:
            element.tail = '\n' + indent * queue[0][0] # for sibling close
        else:
            element.tail = '\n' + indent * (level-1) # for parent close
        queue[0:0] = children # prepend so children come before siblings



if __name__=="__main__":

    # if no argument return 
    if len(sys.argv) < 2:
        print("Input folder argument missing... ")


    if len(sys.argv) == 2:
        input_folder = sys.argv[1]
    
    # get list of filenames in folder
    print(input_folder)
    file_paths = glob.glob(input_folder + '/*.jpg')
    print('Found ', len(file_paths), ' images') 
    # check if list is not empty
    if file_paths:
        # load detection model
        plant_detector = ObjectDetector(DETECTION_MODEL, DETECTION_CATEGORIES)

        # for all elements in file list run object detection 
        for file_path in file_paths:
            # create base xml
            img = cv2.imread(file_path)
            H,W, depth = img.shape
            print(W,H,depth)

            
            basename = os.path.basename(file_path)
            dirname = os.path.dirname(file_path)
            foldername = dirname.split('/')[-1]
            xml_file_path = os.path.join(dirname, basename).replace('.jpg', '.xml')
            print(xml_file_path)

            annotation = ElementTree.Element('annotation')
            ElementTree.SubElement(annotation, 'folder').text = foldername
            ElementTree.SubElement(annotation, 'filename').text = basename
            ElementTree.SubElement(annotation, 'path').text = file_path

            source = ElementTree.SubElement(annotation, 'source')
            ElementTree.SubElement(source, 'database').text = 'Unknown'

            size = ElementTree.SubElement(annotation, 'size')
            ElementTree.SubElement(size, 'width').text = str(W)
            ElementTree.SubElement(size, 'height').text = str(H)
            ElementTree.SubElement(size, 'depth').text = str(depth)

            ElementTree.SubElement(annotation, 'segmented').text = str(0)
            
            resized_img = cv2.resize(img, (W//4, H//4))
            plant_detector.detect(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

            THRESHOLD = 0.5

            for i in range(plant_detector.detections['detection_boxes'].shape[0]):
                if plant_detector.detections['detection_scores'][i] > THRESHOLD:
                    ymin, xmin, ymax, xmax = plant_detector.detections['detection_boxes'][i]
                    detection_class = plant_detector.detections['detection_classes'][i]
                    object_name = plant_detector.category_index[detection_class]['name']
                    #add object to xml
                    xml_object = ElementTree.SubElement(annotation, 'object')
                    ElementTree.SubElement(xml_object, 'name').text = object_name
                    ElementTree.SubElement(xml_object, 'pose').text = 'Unspecified'
                    ElementTree.SubElement(xml_object, 'truncated').text = '0'
                    ElementTree.SubElement(xml_object, 'difficult').text = '0'

                    bndbox = ElementTree.SubElement(xml_object, 'bndbox')
                    ElementTree.SubElement(bndbox, 'xmin').text = str(int(xmin * W))
                    ElementTree.SubElement(bndbox, 'ymin').text = str(int(ymin * H))
                    ElementTree.SubElement(bndbox, 'xmax').text = str(int(xmax * W))
                    ElementTree.SubElement(bndbox, 'ymax').text = str(int(ymax * H))

                    
            # write tree to xml file
            prettify(annotation)
            tree = ElementTree.ElementTree(annotation)
            tree.write(xml_file_path, encoding='UTF-8', xml_declaration=False)

            



            # for every bounding box bigger than threshold save object in xml

















        # xml name is same as filename
        # save xml





    

    
