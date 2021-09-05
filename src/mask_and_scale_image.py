#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
from PIL import Image

    
def convert_png_to_mask(png_mask_name, image_name, scale_percent=100):
    # Convert png output from CVAT annotations to a mask file used for object in the synthesized image
    
    mask = cv.imread(png_mask_name)
    mask = np.sum(mask,2)
    mask2 = np.where((mask>0),0,255).astype('uint8')
    
    cor_image = cv.imread(image_name) # corresponding image
    img_width = cor_image.shape[0] 
    img_height = cor_image.shape[1]
    
    mask_im = Image.fromarray(mask2)
    mask_im.resize((img_width, img_height))
    
    if scale_percent != 100:
        width_mask = int(mask_im.size[0] * scale_percent / 100)
        height_mask = int(mask_im.size[1] * scale_percent / 100)
        dim_mask = (width_mask, height_mask)
        # resize mask
        mask_im = mask_im.resize(dim_mask)
        save_mask_name = image_name.replace('.jpg','_s'+str(int(scale_percent))+'.pbm')
    else: 
        save_mask_name = image_name.replace('.jpg','.pbm')
    
    mask_im.save(save_mask_name)


def convert_png_to_annot_mask(png_annot_mask_name, image_name, scale_percent=100):
    # Convert png output from CVAT annotations to an annotation mask file used to create bounding box for object in the synthesized image
    
    mask = cv.imread(png_annot_mask_name)
    mask = np.sum(mask,2)
    mask2 = np.where((mask>0),0,255).astype('uint8')
    
    cor_image = cv.imread(image_name) # corresponding image
    img_width = cor_image.shape[0] 
    img_height = cor_image.shape[1]
    
    mask_im = Image.fromarray(mask2)
    mask_im.resize((img_width, img_height))
    
    if scale_percent != 100:
        width_mask = int(mask_im.size[0] * scale_percent / 100)
        height_mask = int(mask_im.size[1] * scale_percent / 100)
        dim_mask = (width_mask, height_mask)
        # resize mask
        mask_im = mask_im.resize(dim_mask)
        save_mask_name = image_name.replace('.jpg','_s'+str(int(scale_percent))+'_annot.pbm')
    else: 
        save_mask_name = image_name.replace('.jpg','_annot.pbm')
    
    mask_im.save(save_mask_name)


def replace_image_background(image_name, mask_name):
    # Raplace the image background with the median color of the object. This can prevent unnatural edges from the background when resizing object imagess
    mask = cv.imread(mask_name)
    mask2 = np.where((mask>0),0,1).astype('uint8')
    
    image = cv.imread(image_name) 
    masked_image = image * mask2
    
    two_d_mask =  mask2[:,:,0]
    obj_idx = np.where(two_d_mask==1)
    background_idx = np.where(two_d_mask==0)

    med0 = int(np.median(image[:,:,0][obj_idx]))
    med1 = int(np.median(image[:,:,1][obj_idx]))
    med2 = int(np.median(image[:,:,2][obj_idx]))
    
    transformed_image = image.copy()
    transformed_image[:,:,0][background_idx] = med0
    transformed_image[:,:,1][background_idx] = med1
    transformed_image[:,:,2][background_idx] = med2
    
    
    cv.imwrite(image_name, transformed_image) 


def scale_image(image_name, scale_percent=100):
    # Scale the size of an input image using the given scale percentage
    
    try:
        img = cv.imread(image_name)
    except:
        print("Cannot read image file")   
    
    width_im = int(img.shape[1] * scale_percent / 100)
    height_im = int(img.shape[0] * scale_percent / 100)
    dim_im = (width_im, height_im)
    # resize and save image
    img = cv.resize(img, dim_im, interpolation = cv.INTER_AREA)
    image_name = image_name.replace('.jpg','_s'+str(int(scale_percent))+'.jpg')
    cv.imwrite(image_name, img)
