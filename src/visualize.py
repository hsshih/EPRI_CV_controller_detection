import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import glob
import os
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from src.evaluation import load_image_into_numpy_array, make_detections, \
            postprocess_detections, get_metrics

def read_label_from_xml(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    classes = []
    bboxes = []

    for boxes in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None
        c = boxes.find("name").text
        xmin = int(boxes.find("bndbox/xmin").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        classes.append(c)
        bboxes.append([xmin, ymin, xmax, ymax])

    height = int(root.find("size/height").text)
    width = int(root.find("size/width").text)

    return (classes, bboxes), (height, width)

def plot_train_images_w_labels(src_dir, label_map_path, num_img=6, img_per_row=3, lw=16):

    src_dir = str(src_dir)
    label_map_path = str(label_map_path)
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    images = np.random.choice(glob.glob(os.path.join(src_dir,'*.jpg'))
                            + glob.glob(os.path.join(src_dir,'*.JPG')),
                            num_img, replace=False)
    labels = [os.path.splitext(img)[0]+'.xml' for img in images]
    images = sorted(images)
    labels = sorted(labels)

    num_row = (num_img+img_per_row-1)//img_per_row
    plt.figure(figsize=(13,7*num_row*(2/img_per_row)))

    for n in range(num_img):

        img = cv2.cvtColor(cv2.imread(images[n]), cv2.COLOR_RGB2BGR)
        lbl, _ = read_label_from_xml(labels[n])
        class_names, boxes = lbl
        class_ids = [label_map_dict[class_name] for class_name in class_names]
        boxes = np.array(boxes)
        boxes1 = boxes.copy()
        if len(boxes) != 0:
            # object detection API expects bboxes as [ymin,xmin,ymax,xmax]
            boxes1[:,0] = boxes[:,1]
            boxes1[:,1] = boxes[:,0]
            boxes1[:,2] = boxes[:,3]
            boxes1[:,3] = boxes[:,2]

        dummy_scores = np.ones(len(class_names))
        img_with_labels = img.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            img_with_labels,
            boxes = boxes1,
            classes = class_ids,
            scores = dummy_scores,
            category_index = category_index,
            skip_scores=True,
            line_thickness=lw,
            )

        plt.subplot(num_row, img_per_row, 1+n)
        plt.axis('off')
        plt.imshow(img_with_labels)
        plt.title(os.path.split(images[n])[1])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def plot_train_images_w_labels_from_paths(images, label_map_path, num_img=6, img_per_row=3, lw=16):

    label_map_path = str(label_map_path)
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    images = [str(img) for img in images]
    labels = [os.path.splitext(img)[0]+'.xml' for img in images]
    images =  sorted(images)
    labels = sorted(labels)

    num_row = (num_img+img_per_row-1)//img_per_row
    plt.figure(figsize=(13,7*num_row*(2/img_per_row)))

    for n in range(num_img):

        print(images[n])
        img = cv2.cvtColor(cv2.imread(images[n]), cv2.COLOR_RGB2BGR)
        lbl, _ = read_label_from_xml(labels[n])
        class_names, boxes = lbl
        class_ids = [label_map_dict[class_name] for class_name in class_names]
        boxes = np.array(boxes)
        boxes1 = boxes.copy()
        if len(boxes) != 0:
            # object detection API expects bboxes as [ymin,xmin,ymax,xmax]
            boxes1[:,0] = boxes[:,1]
            boxes1[:,1] = boxes[:,0]
            boxes1[:,2] = boxes[:,3]
            boxes1[:,3] = boxes[:,2]

        dummy_scores = np.ones(len(class_names))
        img_with_labels = img.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            img_with_labels,
            boxes = boxes1,
            classes = class_ids,
            scores = dummy_scores,
            category_index = category_index,
            skip_scores=True,
            line_thickness=lw,
            )

        plt.subplot(num_row, img_per_row, 1+n)
        plt.axis('off')
        plt.imshow(img_with_labels)
        plt.title(os.path.split(images[n])[1])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def draw_predictions_on_images(img_paths, detect_fn, label_map_path, target_size, min_score_thresh=0.5, linewidth = 16):
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
    res_images = []
    for image in tqdm(img_paths):
        image_np = load_image_into_numpy_array(image, target_size)
        detections = postprocess_detections(
            make_detections(image_np, detect_fn),
            min_score_thresh=min_score_thresh
        )
        image_np_with_detections = image_np.copy()

        res_images += [viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            line_thickness=linewidth)]
        
    return res_images


def plot_predictions_from_paths(img_paths, detect_fn, label_map_path,
                                target_size, min_score_thresh=0.5,
                                img_per_row=2, linewidth=16):

    num_img = len(img_paths)
    num_row = (num_img+img_per_row-1)//img_per_row
    res_images = draw_predictions_on_images(img_paths,
                                            detect_fn,
                                            label_map_path,
                                            target_size,
                                            min_score_thresh = min_score_thresh,
                                            linewidth = linewidth)
    plt.figure(figsize=(13,7*num_row*(2/img_per_row)))

    for n in range(num_img):
        plt.subplot(num_row, img_per_row, 1+n)
        plt.axis('off')
        plt.imshow(res_images[n])
        plt.title(os.path.split(img_paths[n])[1])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def save_predictions(img_paths, target_dir, detect_fn, label_map_path, target_size,
                     min_score_thresh=0.5, linewidth=16):
    res_images = draw_predictions_on_images(img_paths,
                                            detect_fn,
                                            label_map_path,
                                            target_size,
                                            min_score_thresh=min_score_thresh,
                                            linewidth=linewidth)
    for i in range(len(res_images)):
        im = Image.fromarray(res_images[i])
        im.save(os.path.join(target_dir,os.path.split(img_paths[i])[1]),"jpeg")


def print_metrics(metrics, image_dir, label_map_path , min_score_thresh=0.5, iou_thresh=0.5):
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    label_map_dict_rev = {value: key for key, value in label_map_dict.items()}
    tot_TP = 0
    tot_FP = 0
    tot_FN = 0
    tot_AP = 0
    for key in label_map_dict.keys():
        tot_TP += metrics[key+'_TP_num']
        tot_FN += metrics[key+'_FN_num']
        tot_FP += metrics[key+'_FP0_num']
        for key2 in label_map_dict_rev.keys():
            if not(key == label_map_dict_rev[key2]):
                tot_FP += metrics[key+'_FP'+str(key2)+'_num']
        tot_AP += metrics['AP_'+key]
    tot_prec = tot_TP/(tot_TP+tot_FP)
    tot_rec  = tot_TP/(tot_TP+tot_FN)
    print('Metrics when evaluated on images in \033[1n{0}\033[0m at minimal score {1} and IoU threshold {2}:\n'
          .format(image_dir, min_score_thresh, iou_thresh))
    for key in label_map_dict.keys():
        print('\033[1m'+key+' AP\033[0m: ', round(metrics['AP_'+key],4))
        print('\033[1m'+key+' Prediction Types\033[0m: \t{0} True Positives,'.format(metrics[key+'_TP_num']))
        print('\t\t\t\t{} False Positives (nothing there),'.format(metrics[key+'_FP0_num']))
        for key2 in label_map_dict_rev.keys():
            if not(key == label_map_dict_rev[key2]):
                print(('\t\t\t\t{} False Positives (mistaken '+label_map_dict_rev[key2]+'),')
                      .format(metrics[key+'_FP'+str(key2)+'_num']))
        print('\t\t\t\t{} False Negatives,'.format(metrics[key+'_FN_num']))
    print('\033[1mTotal Precision\033[0m: {}'.format(round(tot_prec,4)))
    print('\033[1mTotal Recall\033[0m: {}'.format(round(tot_rec,4)))
    print('\033[1mF-Score\033[0m: {}'.format(round(2*tot_prec*tot_rec/(tot_prec+tot_rec),4)))
    print('\033[1mmAP\033[0m: {}'.format(round(tot_AP/len(label_map_dict.keys()),4)))



def plot_detection_confusion_matrix(metrics, label_map_path, title=None, rotate_label=45, save = None):

    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    label_map_dict_rev = {value: key for key, value in label_map_dict.items()}
    category_num = len(label_map_dict.keys())

    plt.figure(figsize=(10,7))
    cm = np.zeros((category_num+1,category_num+1))
    for i in range(category_num):
        for j in range(category_num):
            if i == j:
                cm[i,j] = metrics[label_map_dict_rev[i+1]+'_TP_num']
            else:
                cm[i,j] = metrics[label_map_dict_rev[j+1]+'_FP'+str(i+1)+'_num']
    for i in range(category_num):
        cm[i,category_num] = metrics[label_map_dict_rev[i+1]+'_FN_num']
        cm[category_num, i] = metrics[label_map_dict_rev[i+1] + '_FP0_num']
    cm[category_num,category_num] = float('nan')

    label_list = [key for key in label_map_dict.keys()]+['None']

    df_cm = pd.DataFrame(cm, index = label_list, columns = label_list)
    df_cm.rename_axis("ground truth", axis='rows', inplace=True)
    df_cm.rename_axis("predicted", axis='columns', inplace=True)


    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    sns.set(font_scale=1.2) # for label size
    g = sns.heatmap(df_cm, annot=True, annot_kws={"size": 14}, cmap='Blues', cbar=False)
    g.set_yticklabels(g.get_yticklabels(), rotation = rotate_label)
    g.set_xticklabels(g.get_xticklabels(), rotation = 90 - rotate_label)
    #plt.xticks(rotation=45)
    if title: plt.title(title)
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
