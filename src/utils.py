import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2
import xml.etree.ElementTree as ET
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import xml.dom.minidom

def get_CAM(processed_image, actual_label, model):
    """
    Calculate the gradient-weighted class activation map for an image.
    """

    # locate the last convolutional layer of the model
    for layer in reversed(model.layers):
    # check to see if the layer has a 4D output => convolutional layer
        if len(layer.output.shape) == 4:
            break
    last_conv_layer = layer
    last_conv_num_neurons = layer.output.shape[-1]
    #print(last_conv_layer.name, last_conv_num_neurons)

    model_grad = tf.keras.Model(model.inputs,
                          [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output_values, predictions = model_grad(processed_image)

        # watch the conv_output_values
        tape.watch(conv_output_values)

        ## Use binary cross entropy loss
        ## actual_label is 0 if horse, 1 if human
        # get prediction probability of human
        # If model does well,
        # pred_prob should be close to 0 if horse, close to 1 if human
        pred_prob = predictions[:,0]

        # make sure actual_label is a float, like the rest of the loss calculation
        actual_label = tf.cast(actual_label, dtype=tf.float32)

        # add a tiny value to avoid log of 0
        smoothing = 1e-10

        # Calculate loss as binary cross entropy
        loss = ( actual_label * tf.math.log(pred_prob + smoothing)
            + (1 - actual_label) * tf.math.log(1 - pred_prob + smoothing) )
        #print(f"binary loss: {loss}")

    # get the gradient of the loss with respect to the outputs of the last conv layer
    grads_values = tape.gradient(loss, conv_output_values)
    grads_values = K.mean(grads_values, axis=(0,1,2))

    conv_output_values = np.squeeze(conv_output_values.numpy())
    grads_values = grads_values.numpy()

    # weight the convolution outputs with the computed gradients
    for i in range(last_conv_num_neurons):
        conv_output_values[:,:,i] *= grads_values[i]

    heatmap = np.mean(conv_output_values, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    del model_grad, conv_output_values, grads_values, loss

    return heatmap


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

def write_label_to_xml(xml_file, label, img_shape):

    classes, bboxes = label
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # update image size
    tree.find("size/width").text = str(img_shape[1])
    tree.find("size/height").text = str(img_shape[0])

    # remove objects (number of bboxes after augmentation can be smaller)
    objs = root.findall('object')
    for obj in objs:
        root.remove(obj)

    # add augmented boxes
    for c, bb in zip(classes, bboxes):
        obj = ET.Element('object')
        name = ET.SubElement(obj, 'name')
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        name.text = str(c)
        xmin.text = str(bb[0])
        ymin.text = str(bb[1])
        xmax.text = str(bb[2])
        ymax.text = str(bb[3])
        root.append(obj)

    tree.write(xml_file)
