# -*- coding: utf-8 -*-

import glob
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

def pred_yolo_to_xml(pred_label_dir, image_size=640):
  # Take the predicted labels directory from YOLO v5 output and convert all the predictions to XML files (Pascal VOC format)
  
  label_files = glob.glob('*.txt')

  for label_file in label_files:
    image_name = label_file.replace('.txt','.jpg')
    xml_name =  label_file.replace('.txt','.xml')

    top = Element('annotation')
    
    filename = SubElement(top, 'filename')
    filename.text = image_name
    
    size_root = SubElement(top, 'size')
    width_entry = SubElement(size_root, 'width')
    width_entry.text = '%d'%(image_size)
    height_entry = SubElement(size_root, 'height')
    height_entry.text = '%d'%(image_size)
    depth_entry = SubElement(size_root, 'depth')
    depth_entry.text = '3'

    file1 = open(label_file,"r") 

    while True:
        line = file1.readline().rstrip('\n').split(' ')
        if line[0] == '':
          break

        obj_num = int(line[0])
        if obj_num == 0:
          obj_type = 'AllenBradley'
        if obj_num == 1:
          obj_type = 'Siemens'

        xc = float(line[1])
        yc = float(line[2])
        dx = float(line[3])
        dy = float(line[4])
        conf = float(line[5])

        x = xc * image_size
        y = yc * image_size
        w = dx * image_size
        h = dy * image_size

        xmin = int(x - w/2)
        xmax = int(x + w/2)
        ymin = int(y - h/2)
        ymax = int(y + h/2)

        object_root = SubElement(top, 'object')
        object_type = str(obj_num)
        object_type_entry = SubElement(object_root, 'name')
        object_type_entry.text = obj_type
        object_bndbox_entry = SubElement(object_root, 'bndbox')

        x_min_entry = SubElement(object_bndbox_entry, 'xmin')
        x_min_entry.text = '%d'%(xmin)
        x_max_entry = SubElement(object_bndbox_entry, 'xmax')
        x_max_entry.text = '%d'%(xmax)
        y_min_entry = SubElement(object_bndbox_entry, 'ymin')
        y_min_entry.text = '%d'%(ymin)
        y_max_entry = SubElement(object_bndbox_entry, 'ymax')
        y_max_entry.text = '%d'%(ymax)

        confidence_entry = SubElement(object_root, 'confidence')
        confidence_entry.text = '%.2f'%(conf)

        difficult_entry = SubElement(object_root, 'difficult')
        difficult_entry.text = '0'

    xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
    with open(xml_name, "w") as f:
        f.write(xmlstr)
    
    file1.close()
