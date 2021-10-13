import cv2 
import sys
from pathlib import Path
from lxml import etree
import random

stem_center = None
stem_bbox = None

def set_stem_location(event, x, y, flags, params):
    global stem_center
    global stem_bbox
    stem_box_size = 40

    if event == cv2.EVENT_LBUTTONUP:
        stem_center = (x,y)
        stem_bbox = [(x-stem_box_size//2 + random.randint(-5,5), y-stem_box_size//2 + random.randint(-5,5)),
                     (x+stem_box_size//2 + random.randint(-5,5), y+stem_box_size//2 + random.randint(-5,5))]
        
        #img_roi_copy = img_roi.copy()

        #cv2.rectangle(img_roi_copy, stem_bbox[0], stem_bbox[1], (255,255,0),1)
        #cv2.imshow('img', img_roi_copy)

if __name__=="__main__":
    if len(sys.argv) < 2:
        print('Missing folder path...')
        sys.exit()

    if len(sys.argv) == 2:
        folder_path = Path(sys.argv[1])
        print(folder_path)

    # get xml list from folder path
    xml_list = list(folder_path.glob('**/*.xml'))
    # for xml in list 
        # load xml
    for xml_path in sorted(xml_list):
        print(xml_path)
        with open(xml_path, 'r') as xml_file:
            xml_str = xml_file.read()
        xml = etree.fromstring(xml_str)
        for child in  xml.getchildren():
            if child.tag == 'path':
                img = cv2.imread(child.text)
            if child.tag == 'object':
                for child in child.getchildren():
                    if child.tag == 'name':
                        plant_name = child.text
                    if child.tag == 'bndbox':
                        for coordinate in child.getchildren():
                            if coordinate.tag == 'xmin':
                                xmin = int(coordinate.text)
                            if coordinate.tag == 'ymin':
                                ymin = int(coordinate.text)
                            if coordinate.tag == 'xmax':
                                xmax = int(coordinate.text)
                            if coordinate.tag == 'ymax':
                                ymax = int(coordinate.text)
                img_copy = img.copy()                
                cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0,255,255), 1)
                # work with object
                # calc n pixel around center of bndbox
                delta = 750
                center = (xmin + (xmax-xmin)//2, ymin + (ymax-ymin)//2)
                # roi (xmin, ymin, xmax, ymax)
                roi = (max(0, center[0]-delta//2),
                       max(0, center[1]-delta//2),
                       min(center[0]+delta//2, img.shape[1]),
                       min(center[1]+delta//2, img.shape[0]))
                print(roi[0]- roi[2])
                print(roi[1]-roi[3])
                img_roi=img_copy[roi[1]:roi[3], roi[0]:roi[2]]
                print(img_roi.shape)
            
                #roi=(xmin,xmax, ymin,ymax)
                print(roi)
                #cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (255,0,0), 1 )
                cv2.namedWindow('img')
                cv2.setMouseCallback('img', set_stem_location)
                while True:
                    img_roi_copy = img_roi.copy()
                    #stem_box_size = 40
                    if stem_center:
                        #stem_bbox = [(stem_center[0]-stem_box_size//2, stem_center[1]-stem_box_size//2),
                        #             (stem_center[0]+stem_box_size//2, stem_center[1]+stem_box_size//2)]
                        cv2.rectangle(img_roi_copy, stem_bbox[0], stem_bbox[1], (255,255,0), 1)

                    cv2.imshow('img',img_roi_copy)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('n'):
                        # save stem bbox to xml
                        #cv2.rectangle(img_copy,
                       #               (stem_bbox[0][0]+roi[0], stem_bbox[0][1]+roi[1]),
                       #               (stem_bbox[1][0]+roi[0], stem_bbox[1][1]+roi[1]),
                       #               (255,255,0),
                       #               1)
                       # cv2.imshow('img', img_copy)
                       # cv2.waitKey(0)

                        break
            
                # save stem_bbox to xml
                if stem_bbox:
                    new_object = etree.SubElement(xml,'object')
                    etree.SubElement(new_object, 'name').text = 'beet_stem'
                    etree.SubElement(new_object, 'pose').text = 'Unspecified'
                    etree.SubElement(new_object, 'truncated').text = '0'
                    etree.SubElement(new_object, 'difficult').text = '0'
                    bnd_box = etree.SubElement(new_object, 'bndbox')  
                    etree.SubElement(bnd_box, 'xmin').text = str(stem_bbox[0][0]+roi[0])
                    etree.SubElement(bnd_box, 'ymin').text = str(stem_bbox[0][1]+roi[1])
                    etree.SubElement(bnd_box, 'xmax').text = str(stem_bbox[1][0]+roi[0])
                    etree.SubElement(bnd_box, 'ymax').text = str(stem_bbox[1][1]+roi[1])
                
                    print(etree.tostring(xml,pretty_print=True))
                    stem_bbox = None
                    stem_center = None
       
        et = etree.ElementTree(xml)
        et.write(str(xml_path), pretty_print=True)

                # click on center of stem

                # press n for next image -> save bbox to xml


                                
        #for element in xml.iter():
         #   print(element.tag, element.text)
        # parse xml recursive for bboxes

        # load image

        # show bbox on image
        # click adds stem box of default size



