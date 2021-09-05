import numpy as np
import os
import glob
import shutil
from pathlib import Path
from src.preproc import make_augmented_image_and_label

def clear_dir(dir2clear):
    dir2clear = str(dir2clear)
    if os.path.exists(dir2clear):
        objs = glob.glob(os.path.join(dir2clear, "*"))
        for obj in objs:
            if os.path.isfile(obj):
                os.remove(obj)

def copy_augment_data(
    src_dir, dest_dir,
    class_subdirs = True,
    target_size = 640,
    reshape2square = 'pad',
    no_preproc = False,
    augment_kwargs = {},
    augment_mult = 1):
    """
    Take images and labels from a source directory, resize,
    crop/pad/stretch, and copy them to a destination folder.
    Can also expand data by augmentation.

    Args
        src_dir (str or Path) : source directory
        dest_dir: destination folder where train/validation split is made
        class_subdirs (bool): whether to copy from subfolders of the source
            directory corresponding to different classes (true) or copy from
            the source folder directly (false)
        use_* (bool) : use images from the given source
        target_size : target size of the resulting square images
        reshape2square (str) : method to get square images (pad, crop, or stretch)
        no_preproc : copy images/labels w/o any preprocessing
        augment_kwargs : augmentation params
        augment_mult (int) : increase the number of images via augmentation by this factor
    """

    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    def copy_single_folder(_src_dir, _dest_dir, _data_class):
        images = ( glob.glob(str(_src_dir/'*.jpg'))
                 + glob.glob(str(_src_dir/'*.JPG')) )
        labels = glob.glob(str(_src_dir/'*.xml'))
        images.sort()
        labels.sort()
        # check if the number of images corresponds to the number of .xml files
        assert(len(images)==len(labels))
        n=0
        for image, label in zip(images, labels):
            # make sure the copied images have lowercase extensions
            name_base0 = os.path.splitext(os.path.split(image)[1])[0]
            if no_preproc:
                # no preprocessing, just copy
                shutil.copyfile(image, _dest_dir/(name_base0+'.jpg'))
                shutil.copyfile(label, _dest_dir/(name_base0+'.xml'))
                n += 1
            else:
                # apply augmentation and copy
                for k in range(augment_mult):
                    if k>0: name_base = name_base0 + f"_a{k}"
                    else: name_base = name_base0
                    make_augmented_image_and_label(
                            image, _dest_dir/(name_base+'.jpg'),
                            label, _dest_dir/(name_base+'.xml'),
                            target_max_size = target_size,
                            reshape2square = reshape2square,
                            **augment_kwargs)
                    n += 1
        print(f"Produced {n} images with labels of class {_data_class}")

    if class_subdirs:
        for class_dir in src_dir.iterdir():
            data_class = class_dir.name
            copy_single_folder(class_dir, dest_dir, data_class)
    else:
        data_class = 'all'
        copy_single_folder(src_dir, dest_dir, data_class)


def make_split(src_dir, train_valid_split=0.8):
    """
    Make a train/validation split in src_dir folder.

    Args
        src_dir (str) : folder where the split is done
        train_valid_split (float) : fraction of data in the training split
    Returns
        train_dir, valid_dir
    """

    src_dir = Path(src_dir)
    train_dir = src_dir/'training'
    valid_dir = src_dir/'validation'

    if not train_dir.exists():
        train_dir.mkdir()
    # else:
    #     for f in train_dir.iterdir():
    #         if f.is_file(): f.unlink()
    if not valid_dir.exists():
        valid_dir.mkdir()
    # else:
    #     for f in valid_dir.iterdir():
    #         if f.is_file(): f.unlink()

    images = ( glob.glob(str(src_dir/'*.jpg'))
             + glob.glob(str(src_dir/'*.JPG')) )
    labels = glob.glob(str(src_dir/'*.xml'))
    images.sort()
    labels.sort()
    assert(len(images)==len(labels))
    num_images = len(images)
    num_images_train = int(train_valid_split*num_images)
    indc = np.random.permutation(range(num_images))
    for i in range(num_images_train):
        image = images[indc[i]]
        label = labels[indc[i]]
        shutil.move(image, train_dir)
        shutil.move(label, train_dir)
    for i in range(num_images_train, num_images):
        image = images[indc[i]]
        label = labels[indc[i]]
        shutil.move(image, valid_dir)
        shutil.move(label, valid_dir)

    print("number of training examples:", num_images_train)
    print("number of validation examples:", num_images-num_images_train)
    return (train_dir, valid_dir)


# def combine_augment_data(
#     src_org, src_ggl, src_syn,
#     dest_dir,
#     use_org = True,
#     use_ggl = True,
#     use_syn = True,
#     target_size = 640,
#     reshape2square = 'pad',
#     no_preproc = False,
#     augment_kwargs = None,
#     augment_mult = 1):
#     """
#     Take images and labels from different sources, resize,
#     crop/pad/stretch, and copy them to destination folder.
#     Can also expand data by augmentation.
#
#     Args
#         src_* (str or Path) : source directories
#         dest_dir: destination folder where train/validation split is made
#         use_* (bool) : use images from the given source
#         target_size : target size of the resulting square images
#         reshape2square (str) : method to get square images (pad, crop, or stretch)
#         no_preproc : copy images/labels w/o any preprocessing
#         augment_kwargs : augmentation params
#         augment_mult (int) : increase the number of images via augmentation by this factor
#     """
#
#     dest_dir = Path(dest_dir)
#     src_org = Path(src_org)
#     src_ggl = Path(src_ggl)
#     src_syn = Path(src_syn)
#
#     def resize_copy(_src_dir, _dest_dir, _data_class):
#         images = ( glob.glob(str(_src_dir/'*.jpg'))
#                  + glob.glob(str(_src_dir/'*.JPG')) )
#         labels = glob.glob(str(_src_dir/'*.xml'))
#         images.sort()
#         labels.sort()
#         # check if the number of images corresponds to the number of .xml files
#         assert(len(images)==len(labels))
#         n=0
#         for image, label in zip(images, labels):
#             # apply augmentation function (resize + crop/pad) to images/labels and save
#             # them in 'dest_dir'
#             _dest_img_name = os.path.split(image)[-1].split('.')[0]+'.jpg'
#             if not no_preproc:
#                 make_aug_img_and_lbl(image, _dest_dir/_dest_img_name,
#                                      label, _dest_dir/os.path.split(label)[-1],
#                                      aug_func = aug_resize_crop_pad,
#                                      target_max_size = target_size,
#                                      reshape2square = reshape2square)
#             else:
#                 shutil.copyfile(image, _dest_dir/_dest_img_name)
#                 shutil.copyfile(label, _dest_dir/os.path.split(label)[-1])
#             n += 1
#         print(f"resized and copied {n} images with labels of class {_data_class}")
#
#     if use_org:
#         for class_dir in src_org.iterdir():
#             data_class = class_dir.name
#             resize_copy(class_dir, dest_dir, data_class)
#
#     if use_ggl:
#         for class_dir in src_ggl.iterdir():
#             data_class = class_dir.name
#             resize_copy(class_dir, dest_dir, data_class)
#
#     if use_syn:
#         data_class = 'all'
#         resize_copy(src_syn, dest_dir, data_class)
