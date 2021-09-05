import os, sys
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from src.utils import read_label_from_xml, write_label_to_xml

def resize_crop_pad(src_dir, dest_dir, target_max_size, reshape2square='pad'):

    """
    Resize .jpg images in a given folder, optionally crop or pad to square shape.
    Create a new folder with processed images (original data kept intact).

    Args
      src_dir (string): source folder containing .jpg files to resize
      dest_dir (string): destination folder with resized images
      target_max_size (int): the largest dimension of the resized image
      reshape2square: (string or None):
        None - resize and keep the aspect ratio;
        'stretch' - reshape to square by stetching;
        'crop' - crop to square;
        'pad' - pad to square

    """

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    n=0
    for src in os.listdir(src_dir):
        if os.path.splitext(src)[-1].lower()=='.jpg':

            img = Image.open(os.path.join(src_dir, src))
            img = ImageOps.exif_transpose(img)

            if reshape2square is  None:
                factor = target_max_size / max(img.size)
                img = img.resize((int(factor * img.size[0]),
                                  int(factor * img.size[1])))

            elif reshape2square == 'stretch':
                img = img.resize((target_max_size,target_max_size))

            elif reshape2square=='crop':
                img = ImageOps.fit(img, (target_max_size,target_max_size))

            elif reshape2square=='pad':
                s = max(img.size)
                img = ImageOps.pad(img, (s,s), method=3,
                                 color=None, centering=(0.5, 0.5))
                img = img.resize((target_max_size, target_max_size))

            img.save(os.path.join(dest_dir, src))
            n += 1

    print(f'{n} resized images saved to: {dest_dir}')


def augment_image_array_and_label(img, lbl,
        target_max_size = None,
        reshape2square = 'pad',
        aug_type = None,
        affine_params = None,
        perspect_range = None,
        brightness_range = None,
        contrast_range = None,
        rand_aug_num = 2,
        rand_aug_mag = 1.,
        clip_bb_outside_image = False):
    """
    Apply augmentations to an image array (including resize/crop/pad)
    and modify bounding boxes correspondingly.

    Args
        img (np.array): input image array
        lbl (tuple): (classes, bboxes)
        target_max_size (int): the largest dimension of the resized image, if
            not given, do not resize
        reshape2square (string):
          None - resize and keep the aspect ratio;
          'stretch' - reshape to square by stretching;
          'crop' - crop to square;
          'pad' - pad to square
        rotate range (tuple): random rotation range
        scale_range (tuple): zoom range
        translate_range (tuple): shift range (relative to image size)
        brightness_range (tuple)
        clip_bb_outside_image (bool): clip bounding boxes that end up partially
                                      outside the image
    """
    # Apply 'num' randomly chosen augmentations with magnitude 'mag'.
    #
    #     num (int): number of randomly chosen augmentations
    #     mag (float): magnitude of augmentations (same for all)

    classes, bboxes = lbl
    # form a list of imgaug Bounding Box objects
    _bboxes_im = [BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=c)
                for c,b in zip(classes, bboxes)]
    bboxes_im = BoundingBoxesOnImage(_bboxes_im, shape=img.shape)

    transforms = []

    if reshape2square=='crop':
        h = min(img[:,:,0].shape)
    else:
        h = max(img[:,:,0].shape)
    # resize
    if target_max_size is not None:
        factor = target_max_size / h
        transforms.append(iaa.Resize(factor))
    else:
        target_max_size = h
    # stretch, crop or pad
    if reshape2square=='stretch':
        transforms.append(iaa.Resize((target_max_size, target_max_size)))
    elif reshape2square=='crop':
        transforms.append(iaa.CropToFixedSize(target_max_size, target_max_size,
                                              position='center'))
    elif reshape2square=='pad':
        transforms.append(iaa.PadToFixedSize(target_max_size, target_max_size,
                            pad_mode='constant', position='center'))

    if aug_type=='standard_augment':

        # add contrast variations
        if contrast_range is not None:
            transforms.append(iaa.GammaContrast(contrast_range))
        # add brightness variations
        if brightness_range is not None:
            transforms.append(iaa.Multiply(brightness_range))
        if perspect_range is not None:
            transforms.append(iaa.PerspectiveTransform(scale=perspect_range))
        # add affine transformations
        if affine_params is not None:
            transforms.append(iaa.Affine(**affine_params))

    elif aug_type=='rand_augment':

        mag = rand_aug_mag
        aug_range = (1-0.20*mag, 1+0.20*mag)
        transforms.append(iaa.SomeOf(rand_aug_num,
            [
                iaa.AdditiveGaussianNoise(scale=mag*37, per_channel=True),
                iaa.MotionBlur(k=int(mag*5)),
                iaa.Multiply(aug_range),
                iaa.MultiplyHue(aug_range),
                iaa.MultiplySaturation(aug_range),
                iaa.LinearContrast(aug_range),
                iaa.Sharpen(alpha=0.2*mag, lightness=(0.75, 1.4)),
                iaa.Affine(rotate=(-20*mag, 20*mag)),
                iaa.Affine(shear=(-20*mag, 20*mag)),
            ],
            random_order=True
            )
        )

    # apply transformations to the image and its label
    ia_seq = iaa.Sequential(transforms)
    img_aug, bboxes_im_aug = ia_seq(image=img, bounding_boxes=bboxes_im)

    # remove boxes fully outside of the image after transformations
    bboxes_im_aug = bboxes_im_aug.remove_out_of_image()
    # clip those partially outside
    if clip_bb_outside_image:
        bboxes_im_aug = bboxes_im_aug.clip_out_of_image()
    classes = [bb.label for bb in bboxes_im_aug.items]

    bboxes_aug = list(bboxes_im_aug.to_xyxy_array(dtype=np.int32))
    return img_aug, (classes, bboxes_aug)

def make_augmented_image_and_label(
            img_file, img_file_dst,
            lbl_file, lbl_file_dst,
            **aug_kwargs):
    """
    Generate a new image and label by augmenting an existing one.

    Args
        img_file (str): path to the source image
        img_file_dst (str): path to the new image
        lbl_file (str): path to the source label as an .xml file
        lbl_file_dest (str): path to the new image label
        aug_kwargs: arguments for the augmentation function
    """
    img = cv2.imread(str(img_file))
    lbl, _ = read_label_from_xml(str(lbl_file))
    img_aug, lbl_aug = augment_image_array_and_label(img, lbl, **aug_kwargs)

    cv2.imwrite(str(img_file_dst), img_aug)
    shutil.copyfile(str(lbl_file), lbl_file_dst)
    write_label_to_xml(str(lbl_file_dst), lbl_aug, img_aug.shape)


def split_image_to_squares(img, lbl=None):

    bboxes_im = None
    if lbl is not None:
        classes, bboxes = lbl
        _bboxes_im = [BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=c)
                    for c,b in zip(classes, bboxes)]
        bboxes_im = BoundingBoxesOnImage(_bboxes_im, shape=img.shape)

    min_side = min(img[:,:,0].shape)
    # print(min_side)
    if img.shape[0]>img.shape[1]:
        transform0 = iaa.CropToFixedSize(min_side, min_side,
                                        position='center-bottom')
        transform1 = iaa.CropToFixedSize(min_side, min_side,
                                        position='center-top')
        x1, y1 = 0, img.shape[0] - min_side
        # print('1',x1,y1)
    else:
        transform0 = iaa.CropToFixedSize(min_side, min_side,
                                        position='right-center')
        transform1 = iaa.CropToFixedSize(min_side, min_side,
                                        position='left-center')
        x1, y1 = img.shape[1] - min_side, 0
        # print('2',x1,y1)

    # apply transformations to the image and its label
    img_aug0, bboxes_im_aug0 = transform0(image=img, bounding_boxes=bboxes_im)
    img_aug1, bboxes_im_aug1 = transform1(image=img, bounding_boxes=bboxes_im)

    if lbl is not None:
        bboxes_im_aug0 = bboxes_im_aug0.remove_out_of_image()
        bboxes_im_aug1 = bboxes_im_aug1.remove_out_of_image()
        classes0 = [bb.label for bb in bboxes_im_aug0.items]
        classes1 = [bb.label for bb in bboxes_im_aug1.items]
        bboxes_aug0 = list(bboxes_im_aug0.to_xyxy_array(dtype=np.int32))
        bboxes_aug1 = list(bboxes_im_aug1.to_xyxy_array(dtype=np.int32))
    else:
        bboxes_aug0,bboxes_aug1, classes0,classes1 = None,None, None,None

    return img_aug0, img_aug1, (x1,y1), (classes0, bboxes_aug0), (classes1, bboxes_aug1)

def make_split_images_and_labels(src_dir, dest_dir, only_top_left=False):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for obj in os.listdir(src_dir):
        if os.path.splitext(obj)[1].lower()=='.jpg':

            name = os.path.splitext(obj)[0]
            img_file = os.path.join(src_dir, obj)
            lbl_file = os.path.join(src_dir, name+'.xml')
            img_file_dst0 = os.path.join(dest_dir, name+'_0.jpg')
            lbl_file_dst0 = os.path.join(dest_dir, name+'_0.xml')
            img_file_dst1 = os.path.join(dest_dir, name+'_1.jpg')
            lbl_file_dst1 = os.path.join(dest_dir, name+'_1.xml')

            img = cv2.imread(img_file)
            # print('img_shape', img.shape)
            lbl, _ = read_label_from_xml(lbl_file)
            img_aug0, img_aug1, pos1, lbl_aug0, lbl_aug1 = split_image_to_squares(img, lbl)
            # print('img0_shape', img_aug0.shape)
            # print('img1_shape', img_aug1.shape)

            cv2.imwrite(img_file_dst0, img_aug0)
            shutil.copyfile(lbl_file, lbl_file_dst0)
            write_label_to_xml(lbl_file_dst0, lbl_aug0, img_aug0.shape)
            if not only_top_left:
                cv2.imwrite(img_file_dst1, img_aug1)
                shutil.copyfile(lbl_file, lbl_file_dst1)
                write_label_to_xml(lbl_file_dst1, lbl_aug1, img_aug1.shape)
