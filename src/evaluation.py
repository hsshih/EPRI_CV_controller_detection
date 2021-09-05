import numpy as np
import math, os, shutil, glob
import bisect
import collections
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

from object_detection.utils import label_map_util
from object_detection.metrics import coco_tools
from object_detection.utils import np_box_ops as npops


def load_image_into_numpy_array(image_path, target_size, padding_flag = False):
    """
    Load an image for testing/evaluation into a numpy array. If target_size
    is given, then pad and resize. Also corrects for the EXIF rotation.
    
    Args
        image_path (str): path to the image used for testing/evaluation
        target_size (int): size of the final resized image
            - if target_size is None: the images get passed to the models as is (as rectangles)
            - if target_size is set: the images get resized passed to to a (target_size, target_size)-square
        padding_flag (bool): whether the image is padded or not
            - if target_size is set and padding_flag is False: images get resized and stretched to a (target_size, target_size)-square
            - if target_size is set and padding_flag is True: images get resized to a (target_size,target_size)-square and padded (instead of stretched) to keep original aspect ratio
        
    Return:
        image (np.array): a numpy array of the loaded image
    """
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    # remove alpha channel (4th dimension) for cropped test images
    if np.array(image).shape[2]==4:
      image = Image.fromarray(np.array(image)[:, :, 0:3])

    if target_size is not None:
        s = max(image.size)
        if padding_flag:
            image = ImageOps.pad(image, (s,s))
        return np.array(image.resize((target_size,target_size)))
    else:
        return np.array(image)

def make_detections(image_np, detect_fn):
    """ 
    Make detections on an image numpy array using a detection model. 
    """
    return detect_fn(tf.convert_to_tensor(image_np)[tf.newaxis, ...])

def postprocess_detections(detections_tf, min_score_thresh=0.5):
    """
    Preprocess raw TensorFlow detections by converting them to numpy arrays
    and removing low-confidence predictions.
    
    Args
        detections_tf ():
        min_score_thresh (float): confidence level below which predictions are discarded
    """
    num_detections = len([x for x in detections_tf['detection_scores'].numpy()[0]
                          if x>min_score_thresh])
    detections_tf.pop('num_detections')
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections_tf.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detections['num_detections'] = num_detections
    return detections

def detections_from_image(image_path, detect_fn, target_size, padding_flag, min_score_thresh=0.5):
    """
    Make postprocessed detections for an image given its path and a detection model.
    
    Args
        image_path (str): path to the image used for testing/evaluation
        detect_fn (tf.saved_model): tensorflow loaded model 
        target_size (int): size of the final resized image (see argument of load_image_into_numpy_array())
        padding_flag (bool): whether the image is padded or not (see argument of load_image_into_numpy_array())
        min_score_thresh (float): confidence level below which predictions are discarded
            
    """
    image_np = load_image_into_numpy_array(image_path, target_size, padding_flag)
    detections = make_detections(image_np, detect_fn)
    return postprocess_detections(detections, min_score_thresh)

def detections_to_xml(path_to_xml_file, detections,
                      image_path, label_map_path,
                      image_shape,
                      write_normalized_coordinates = False):
    """
    Save a detections dictionary as an XML file.

    Args
        path_to_xml_file (str): path to output the XML file
        detections (dict): preprocessed dictionary of detections
        image_path (str): path to the image used for inference
        image_size (tuple): shape of the image used for inference
        label_map_path (str): path to the label map
        write_normalized_coordinates (bool): whether write normalized or pixel
                                        coordinates of bounding boxes
    """

    # Assume detection scores are sorted in descending order.

    category_index = label_map_util.create_category_index_from_labelmap(str(label_map_path))

    root = ET.Element("detections")
    image_path = str(image_path)
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    ET.SubElement(root, "folder").text = os.path.dirname(image_path)
    ET.SubElement(root, "path").text = image_path

    # ET.SubElement(root, "num_detections").text = str(num_detections)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_shape[1])
    ET.SubElement(size, "height").text = str(image_shape[0])
    num_objects = len(detections["detection_classes"])

    for n in range(num_objects):
        obj = ET.SubElement(root, "object")
        name = category_index[detections['detection_classes'][n]]['name']
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "confidence").text = str(detections['detection_scores'][n])
        box = ET.SubElement(obj, "bndbox")
        ymin,xmin, ymax,xmax = detections['detection_boxes'][n]
        if not write_normalized_coordinates:
            xmin = int(xmin*image_shape[1])
            ymin = int(ymin*image_shape[0])
            xmax = int(xmax*image_shape[1])
            ymax = int(ymax*image_shape[0])
        ET.SubElement(box, "xmin").text = str(xmin)
        ET.SubElement(box, "ymin").text = str(ymin)
        ET.SubElement(box, "xmax").text = str(xmax)
        ET.SubElement(box, "ymax").text = str(ymax)

    tree = ET.ElementTree(root)
    tree.write(path_to_xml_file)

def detections_from_xml(path_to_xml_file, label_map_path, padding_flag=False,
                        read_normalized_coordinates = False,
                        flag = 'predict'):
    """
    Read detections from an XML file.

    Args
        path_to_xml_file (str): path to the XML
        label_map_path (str): path to the label map
        flag (str):
            'gt': read ground truth boxes/classes;
            'predict': read predicted boxes/classes and confidence scores
    Return:
        objects (dict): dictionary of detections with keys 'detection_boxes',
                'detection_classes', and 'detection_scores' (if 'predict')
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    tree = ET.parse(path_to_xml_file)
    root = tree.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    if padding_flag:
        pad_length = abs(width-height)*0.5
        s = max(width, height)

    if flag == 'gt':
        objects = {'detection_boxes': [], 'detection_classes': []}
    elif flag == 'predict':
        objects = {'detection_boxes': [], 'detection_classes': [], 'detection_scores': []}

    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        ymin = bndbox.find('ymin').text
        xmin = bndbox.find('xmin').text
        ymax = bndbox.find('ymax').text
        xmax = bndbox.find('xmax').text

        if read_normalized_coordinates:
            value = [ float(ymin), float(xmin), float(ymax), float(xmax) ]
        else:
            if padding_flag:
                if width > height:
                    value = [ (int(ymin)+pad_length)/s, int(xmin)/s,
                              (int(ymax)+pad_length)/s, int(xmax)/s ]
                elif height >= width:
                    value = [ int(ymin)/s, (int(xmin)+pad_length)/s,
                              int(ymax)/s, (int(xmax)+pad_length)/s ]
            else:
                value = [ int(ymin)/height, int(xmin)/width,
                          int(ymax)/height, int(xmax)/width ]
        name_code = label_map_dict[member.find('name').text]
        objects['detection_boxes'].append(value)
        objects['detection_classes'].append(name_code)
        if flag == 'predict':
            score = member.find('confidence').text
            objects['detection_scores'].append(score)
    for key in objects.keys():
        objects[key] = np.array(objects[key])
    objects['detection_boxes'] = objects['detection_boxes'].astype('float32')
    objects['detection_classes'] = objects['detection_classes'].astype('int64')
    return objects

def count_fn(true_fn_flags, ground_truth_classes, class_code):
    """
    Count how many true FN are left in true_fn_flags for class given by class_code

    Args
        true_fn_flags: list of flags that are left 1 if ground truth has not been detected at all
        ground_truth_classes: list of classes corresponding to the ground truths in true_fn_flags
        class_code: code of class that is of interest
    Returns:
        number of 1s left true_fn_flags that correspond to class given by class_code
    """
    count = 0
    for i in range(len(true_fn_flags)):
        if true_fn_flags[i] == 1 and ground_truth_classes[i] == class_code:
            count += 1

    return count

def delete_low_score_xml_predictions(predictions, min_score_thresh):
    index = bisect.bisect_left(predictions['detection_scores'].astype('float'), min_score_thresh)
    return {key: value[index:] for key, value in predictions.items()}

def get_tp_fp_for_image(image_path,
                        detect_fn,
                        label_map_path,
                        target_size,
                        padding_flag,
                        min_score_thresh=0.5,
                        iou_thresh=0.5,
                        predictions_source = 'image',
                        pred_xml_dir = None):
    """
    Classify predictions as True/False Positive for an image given its path.

    Args
        image_path (str): path to the evaluated image
        detect_fn: detection model
        target_size: input size of the image before making predictions
        label_map_path: path to the label map file
        padding_flag: choose whether or not images should get padded or stretched to reach target_size
        min_score_thresh (float): confidence level below which predictions are discarded
        iou_thresh (float): IOU threshold to identify a matching prediction
        predictions_source (str):
            'image': run detection on the image to get predictions;
            'xml': read predictions from an XML file with the same name, located
                   in 'pred_xml_dir' folder
        pred_xml_dir (str): path to folder containing predictions as XML files
    Returns:
        predictions_status (defaultdict): dictionary that assigns a detection
            score, predicted class and TP/FP label to each valid prediction;
        gt_num_AB (int): number of ground truth boxes for Allen Bradley;
        gt_num_S (int):  number of ground truth boxes for Siemens;
    """

    # path to the XML file corresponding to the image at 'image_path'
    gt_xml_path = os.path.splitext(image_path)[0]+'.xml'
    ground_truths = detections_from_xml(gt_xml_path, label_map_path, padding_flag = padding_flag, flag='gt')

    if predictions_source == 'image':
        predictions = detections_from_image(image_path, detect_fn, target_size, padding_flag,
                                            min_score_thresh)
    elif predictions_source == 'xml':
        pred_xml_path = os.path.join(pred_xml_dir, os.path.basename(gt_xml_path))
        if Path(pred_xml_path).exists():
            predictions = detections_from_xml(pred_xml_path, label_map_path,
                                              flag='predict')
            predictions = delete_low_score_xml_predictions(predictions, min_score_thresh)
        else:
            predictions = {'detection_boxes': np.zeros((0, 4)).astype('float32'),
                           'detection_classes': np.array([]).astype('int64'),
                           'detection_scores': np.array([]).astype('float32')}
    image_name = os.path.split(image_path)[1]
    ious = npops.iou(ground_truths['detection_boxes'],predictions['detection_boxes'])
    ground_truth_flags = np.ones(np.shape(ious)[0])
    true_fn_flags = np.ones(np.shape(ious)[0]).astype('int')
    predictions_status = collections.defaultdict(list)
    for prediction in range(np.shape(ious)[1]):
        tmp_flag = 0
        gt_flag = 0
        for ground_truth in range(np.shape(ious)[0]):
            if ious[ground_truth,prediction] > iou_thresh:
                if ground_truths['detection_classes'][ground_truth] == predictions['detection_classes'][prediction] \
                and ground_truth_flags[ground_truth] == 1:
                    tmp_flag += 1
                    ground_truth_flags[ground_truth] -= 1
                    true_fn_flags[ground_truth] -= 1
                else:
                    gt_flag = ground_truths['detection_classes'][ground_truth]
                    true_fn_flags[ground_truth] -= 1
        if tmp_flag == 1:
            predictions_status[image_name+'_pred_'+str(prediction)] = \
                        [predictions['detection_scores'][prediction], \
                         predictions['detection_classes'][prediction],'TP']
        elif tmp_flag == 0:
            predictions_status[image_name+'_pred_'+str(prediction)] = \
                        [predictions['detection_scores'][prediction], \
                         predictions['detection_classes'][prediction],'FP'+str(gt_flag)]
        else:
            print('WARNING! There are most likely overlapping boxes and non-max suppression is not working at IOU thresh: ',
                  iou_thresh)
    if any(x<0 for x in ground_truth_flags):
        print('WARNING! Same object has served as ground truth for multiple TP!')
    category_num = len(label_map_util.get_label_map_dict(label_map_path))
    return (predictions_status,
            [list(ground_truths['detection_classes']).count(x+1) for x in range(category_num)],
            [count_fn(true_fn_flags,ground_truths['detection_classes'], x+1) for x in range(category_num)])

def recall_v_precision(predictions_status, ground_truth_num):
    """
    Classify predictions as True/False Positive for an image given its path.

    Args
        predictions_status: list of predictions classified with TP/FP label
        for all images of interest which have been sorted by prediction scores
        ground_truth_num: number of total groundtruths
    Returns:
        rec_v_prec: list of 2-elements lists consisting of recall v predictions
        points along the path of sorted predictions
    """
    rec_v_prec = []
    tp_num = 0
    fp_num = 0
    for ps in predictions_status:
        if ps[2] == 'TP':
            tp_num += 1
        elif ps[2][:2:] == 'FP':
            fp_num += 1
        rec_v_prec.append([tp_num/ground_truth_num, tp_num/(tp_num+fp_num)])
    return np.array(rec_v_prec)

def AP(rec_v_prec):
    """
    Calculates 101-interpolated Average Precision given recall v precision curve

    Args
        rec_v_prec: list of 2-elements lists consisting of recall v predictions
        points along the path of sorted predictions
    Returns:
        AP: Average Precision
    """
    AP = 0
    for i in np.arange(101)/100:
        ind = bisect.bisect_left(rec_v_prec[:,0],i)
        if ind < len(rec_v_prec[:,0]):
            AP += np.max(rec_v_prec[ind::,1])
    return AP/101

def get_metrics(image_dir,
                detect_fn,
                label_map_path,
                target_size,
                padding_flag = False,
                min_score_thresh=0.5,
                iou_thresh=0.5,
                predictions_source = 'image',
                pred_xml_dir = None):
    """
    Calculate evaluation metrics for images in a given folder.
    
    Args
        image_dir (str): path to the dir of images to calculate evaluation metrics
        detect_fn (tf.saved_model) : detection model
        label_map_path: path to the label map file
        target_size: input size of the images
        padding_flag: choose whether or not images should get padded or stretched to reach target_size (see load_image_into_numpy_array())
        min_score_thresh (float): confidence level below which predictions are discarded
        iou_thresh (float): IOU threshold to identify a matching prediction
        predictions_source (str):
            'image': run detection on the image to get predictions;
            'xml': read predictions from an XML file with the same name, located
                   in 'pred_xml_dir' folder
        pred_xml_dir (str): path to folder containing predictions as XML files
        
    Return:
        metrics (dict): metrics
    
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    label_map_dict_rev = {value: key for key, value in label_map_dict.items()}
    ground_truths_nums = np.zeros(len(label_map_dict_rev.keys()))
    true_fn_nums = np.zeros(len(label_map_dict_rev.keys()))

    predictions = collections.defaultdict(list)
    recall_v_precisions = collections.defaultdict(list)
    counters = {}
    images = [x for x in os.listdir(image_dir) if (('.jpg' in x) or ('.JPG' in x))]

    for image in tqdm(images):
        tmp_metric, tmp_tot_nums, tmp_fn_nums = get_tp_fp_for_image(
                    os.path.join(image_dir,image),
                    detect_fn,
                    label_map_path,
                    target_size,
                    padding_flag,
                    min_score_thresh,
                    iou_thresh,
                    predictions_source,
                    pred_xml_dir)

        for value in tmp_metric.values():
            predictions[label_map_dict_rev[value[1]]].append(value)
        ground_truths_nums += tmp_tot_nums
        true_fn_nums += tmp_fn_nums

    tmp_count = 0
    for key in label_map_dict.keys():
        predictions[key] = sorted(predictions[key], reverse = True, key = lambda x: x[0])
        recall_v_precisions[key] = recall_v_precision(predictions[key], ground_truths_nums[tmp_count])
        tmp_count += 1
        counters[key] = collections.Counter(np.array(predictions[key]).flatten())

    metrics = collections.defaultdict(int)
    tmp_count = 0
    for key in label_map_dict.keys():
        metrics['AP_'+key] = AP(recall_v_precisions[key])
        metrics[key+'_TP_num'] = counters[key]['TP']
        metrics[key+'_FN_num'] = int(true_fn_nums[tmp_count])
        tmp_count += 1
        metrics[key+'_FP0_num'] = counters[key]['FP0']
        for key2 in label_map_dict_rev.keys():
            if not(key == label_map_dict_rev[key2]):
                metrics[key+'_FP'+str(key2)+'_num'] = counters[key]['FP'+str(key2)]
    return metrics

def check_if_detection_from_xml(xml_path, subcategory=None):
    """
    Check if a .xml file contains any objects of given category - needed to check
    if image contains any groundtruths.
    """
    objects = ET.parse(xml_path).getroot().findall('object')
    if subcategory is not None:
        objects = [x for x in objects if x.find('name').text==subcategory]
    return bool(len(objects))

def create_img_ids_dict(gt_dir, subcategory=None):
    """
    Create dictionary assigning boolean of whether groundtruths are present to each image id. This
    is required for the COCO metrics API.

    Args:
        gt_dir: path to folder containing groundtruth .xml files
        subcategory: if given only count images that contain groundtruths of that category
        as 'True'
    Returns:
        image_ids: dictionary assigning bools to image ids in image_dir
    """
    image_ids = {}
    for image in os.listdir(gt_dir):
        if ('.jpg' in image) or ('.JPG' in image):
            image_ids[os.path.splitext(image)[0]] = check_if_detection_from_xml(
                    os.path.join(gt_dir, os.path.splitext(image)[0]+'.xml'), subcategory)
    return image_ids

def create_groundtruth_list(gt_dir, label_map_path, target_size, padding_flag, subcategory=None):
    """
    Create groundtruth_list required by COCO metrics API from groundtruth .xml.

    Args:
        gt_dir: path to folder containing groundtruth .xml files
        label_map_path: path to .pbtxt containing label map information
        subcategory: if given will only add groundtruths matching given category
    Returns:
        groundtruth_list: list of dictionaries for each groundtruth containing
        relevant data to calculate COCO metrics
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    groundtruth_list = []
    id_count = 1
    for xml in os.listdir(gt_dir):
        if '.xml' in xml:
            root = ET.parse(os.path.join(gt_dir, xml)).getroot()
            if target_size is not None:
                width = int(root.find('size').find('width').text)
                height = int(root.find('size').find('height').text)
                wf = target_size / width
                hf = target_size / height
            else:
                wf = 1
                hf = 1
            if padding_flag:
                pad_length = abs(width - height) * 0.5
                s = max(width, height)
                wf = target_size / s
                hf = target_size / s

            for member in root.findall('object'):
                bndbox = member.find('bndbox')
                if padding_flag:
                    if width > height:
                        value = [
                            int(bndbox.find('xmin').text) * wf,
                            (int(bndbox.find('ymin').text) + pad_length) * hf,
                            (int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)) * wf,
                            (int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)) * hf]
                    elif height >= width:
                        value = [
                            (int(bndbox.find('xmin').text) + pad_length) * wf,
                            int(bndbox.find('ymin').text) * hf,
                            (int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)) * wf,
                            (int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)) * hf]
                else:
                    value = [
                        int(bndbox.find('xmin').text) * wf,
                        int(bndbox.find('ymin').text) * hf,
                        (int(bndbox.find('xmax').text)-int(bndbox.find('xmin').text)) * wf,
                        (int(bndbox.find('ymax').text)-int(bndbox.find('ymin').text)) * hf]
                class_name=member.find('name').text
                if subcategory is not None:
                    if subcategory == class_name:
                        name_code = label_map_dict[class_name]
                        groundtruth_list.append({'id': id_count,
                                                 'image_id': os.path.splitext(xml)[0],
                                                 'category_id': name_code,
                                                 'bbox': value,
                                                 'area': value[2] * value[3],
                                                 'iscrowd': False
                                                 })
                        id_count += 1
                else:
                    name_code = label_map_dict[member.find('name').text]
                    groundtruth_list.append({'id': id_count,
                                             'image_id': os.path.splitext(xml)[0],
                                             'category_id': name_code,
                                             'bbox': value,
                                             'area': value[2]*value[3],
                                             'iscrowd': False
                                             })
                    id_count += 1
    return groundtruth_list

def convert_bbox_coords(bbox_coords, height, width):
    # incoming format: [ymin, xmin, ymax, xmax] normalized
    # needed format: [xmin, ymin, xwidth, yheight] not normalized
    return [bbox_coords[1]*width,
            bbox_coords[0]*height,
            (bbox_coords[3]-bbox_coords[1])*width,
            (bbox_coords[2]-bbox_coords[0])*height
           ]

def create_detection_boxes_list_from_detect_fn(image_dir, detect_fn, target_size, padding_flag):
    """
    Create detection_boxes_list required by COCO metrics API from a detect_fn

    Args:
        image_dir: path to folder containing images to predict on
        detect_fn: TF 2 detection function loaded from saved model or checkpoint
        target_size: target size for image resizing
    Returns:
        detection_boxes_list: list of dictionaries for each prediction containing
        relevant data to calculate COCO metrics
    """
    detection_boxes_list = []
    images = [x for x in os.listdir(image_dir) if (('.jpg' in x) or ('.JPG' in x))]

    for image in tqdm(images):
        image_np = load_image_into_numpy_array(os.path.join(image_dir, image),
                                               target_size = target_size,
                                               padding_flag = padding_flag)
        height, width, channels = np.shape(image_np)

        detections = postprocess_detections(
            make_detections(image_np, detect_fn),
            min_score_thresh=0.
        )
        for i in range(detections['num_detections']):
            detection_boxes_list.append({
                'image_id': os.path.splitext(image)[0],
                'category_id': detections['detection_classes'][i],
                'bbox': convert_bbox_coords(detections['detection_boxes'][i], height, width),
                'score': detections['detection_scores'][i]
            })

    return detection_boxes_list

def create_detection_boxes_list_from_xml(image_dir, label_map_path):
    """
    Create detection_boxes_list required by COCO metrics API from a detect_fn

    Args:
        image_dir: path to folder containing xml files containing predictions
        label_map_path: path to .pbtxt containing label map information
    Returns:
        detection_boxes_list: list of dictionaries for each prediction containing
        relevant data to calculate COCO metrics
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    detection_boxes_list = []

    for xml in tqdm(os.listdir(image_dir)):
        if ('.xml' in xml) or ('.XML' in xml):
            root = ET.parse(os.path.join(image_dir, xml)).getroot()
            for member in root.findall('object'):
                bndbox = member.find('bndbox')
                value = [
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                    int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)]
                name_code = label_map_dict[member.find('name').text]
                score = member.find('confidence').text

                detection_boxes_list.append({
                    'image_id': os.path.splitext(xml)[0],
                    'category_id': name_code,
                    'bbox': value,
                    'score': score
                })

    return detection_boxes_list

def create_groundtruth_dict(gt_dir, label_map_path, target_size, padding_flag, subcategory=None):
    """
    Create groundtruth dictionary from .xml groundtruths and .pbtxt label map
    """

    image_ids = create_img_ids_dict(gt_dir, subcategory)
    category_index = label_map_util.create_category_index_from_labelmap(
        label_map_path, use_display_name=True)
    categories = [x for x in category_index.values()]
    if subcategory is not None:
        categories = [x for x in categories if x['name']==subcategory]
    groundtruth_list = create_groundtruth_list(gt_dir, label_map_path, target_size, padding_flag, subcategory)

    groundtruth_dict = {
        'annotations': groundtruth_list,
        'images': [{'id': image_id} for image_id in image_ids],
        'categories': categories
    }
    return groundtruth_dict

def get_coco_metrics_from_gt_and_det(groundtruth_dict, detection_boxes_list, category=''):
    """
    Get COCO metrics given dictionary of groundtruth dictionary and the list of
    detections.
    """
    coco_wrapped_groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(detection_boxes_list)
    box_evaluator = coco_tools.COCOEvalWrapper(coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)
    box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
        include_metrics_per_category=False,
        all_metrics_per_category=False,
        super_categories=None
    )
    box_metrics.update(box_per_category_ap)
    box_metrics = {'DetectionBoxes_'+ category + key: value
                   for key, value in iter(box_metrics.items())}
    return box_metrics

def filter_det_list(det_list, label_map_dict, subcategory):
    """
    Filter out those detections that do not belong to subcategory.
    """
    return [x for x in det_list if x['category_id']==label_map_dict[subcategory]]

def get_coco_metrics(gt_dir,
                     label_map_path,
                     target_size,
                     padding_flag,
                     detect_fn = None,
                     image_dir = None,
                     include_metrics_per_category = False):
    """
    Get COCO metrics given a folder with images and ground truth annotations,
    a detection model or prediction-xml-files and a label map.

    Args:
        gt_dir: path to folder containing groundtruth .xml files
        label_map_path: path to .pbtxt containing label map information
        target_size: if detect_fn mode is chosen and target_size is supplied, images will be
        padded to square and resized to (target_size,target_size) before evaluation
        detect_fn: a TF 2 detection function loaded from saved model or checkpoint
        image_dir: a folder containing .xml files with prediction for images
        include_metrics_per_category: if True, COCO metrics will be calculated for each category

    Returns:
        Dictionary containing all the COCO metrics - which are also printed during calculation.
    """
    if (detect_fn == None) and (image_dir == None):
        raise ValueError('Need either a detect function or .xml-prediction directory')
    if (detect_fn is not None) and (image_dir is not None):
        raise ValueError('Supply only one of detect_fn or image_dir, not both!')

    if detect_fn is not None:
        det_list = create_detection_boxes_list_from_detect_fn(gt_dir, detect_fn, target_size, padding_flag)
    elif image_dir is not None:
        det_list = create_detection_boxes_list_from_xml(image_dir, label_map_path)

    if include_metrics_per_category:
        coco_metrics = {}
        label_map_dict = label_map_util.get_label_map_dict(label_map_path)
        for subcategory in label_map_dict.keys():
            gt_dict = create_groundtruth_dict(gt_dir, label_map_path, target_size, padding_flag, subcategory)
            det_list_tmp = filter_det_list(det_list, label_map_dict, subcategory)
            coco_metrics.update(get_coco_metrics_from_gt_and_det(gt_dict,det_list,category=subcategory+'_'))
        gt_dict = create_groundtruth_dict(gt_dir, label_map_path, target_size, padding_flag)
        coco_metrics.update(get_coco_metrics_from_gt_and_det(gt_dict, det_list))
        return coco_metrics
    else:
        gt_dict = create_groundtruth_dict(gt_dir, label_map_path, target_size, padding_flag)
        return get_coco_metrics_from_gt_and_det(gt_dict, det_list)