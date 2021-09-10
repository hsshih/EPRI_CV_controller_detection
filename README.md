# EPRI-CV

This is a proof of concept project that trains object detection algorithms to create inventories of controllers from images of electrical cabinets. We've generated synthetic training images to overcome the lack of adequate training data. The result of this project is presented in [EPRI-CV-results.pdf](https://github.com/hsshih/EPRI_CV_controller_detection/blob/main/EPRI-CV-results.pdf)


## Pre-requisites 

1. Tensorflow
2. The Tensorflow object detection API

A good example on how to do these two steps is given on https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html. This tutorial also shows how to install GPU Support.


## Assumed project structure:

```
main_project_dir/
├─ README.md                               <- The top-level README you are currently reading.
├─ generate-synthetic-images/              <- the scripts to generate synthetic images that will serve as training
├─ data/                  
│   ├─ test-annotated-images/              <- test images and the xml files with their labels
│   ├─ synthetic-annotated-train-images/   <- synthetic images used for training and the xml files with their labels
│   ├─ train-valid-split
│   │    ├─ training/                      <- training images and the xml with their labels used for training
│   │    ├─ validation/                    <- validation images and the xml with their labels used for training
│   │    ├─ train.tfrec                    <- tfrecord of the training images and labels
│   │    ├─ valid.tfrec                    <- tfrecord of the validation images and labels
│   │    └─ label_map.pbtxt                <- label map for this training/validation split
│   └─ ...                                 <- (other train-valid-splits, for example)
├─ models/                              
│   ├─ fine-tuned/                         <- fine-tuned models, obtained from re-training with the train-valid-splits
│   └─ pre-trained/                        <- pre-trained model(s) downloaded from the Tensorflow object detection API zoo
├─ notebooks/                              <- notebooks used for training and evaluation
│   ├─ EPRI_ObjectDetection_Training.ipynb
│   └─ EPRI_ObjectDetection_Evaluation.ipynb
└─ src/                                    <- scripts needed in the notebooks
```


## Labeling the images

In order to train the model, labels should be added to the train and test images. One way to do so manually is with the `labelImg` tool.

It can be installed very easily with a basic `pip install labelImg` command. A nice tutorial on how to use this tool is given here: https://medium.com/deepquestai/object-detection-training-preparing-your-custom-dataset-6248679f0d1d. This will generate, for each image, an .xml file with the same name as the original image. These .xml files contain the bounding boxes and the labels of the objects that are present on the image.

Once the images and their labels are ready and in their right folder, we can go for training.

## Training a model from the Tensorflow API zoo

The EPRI_ObjectDetection_Training.ipynb notebook allows for the training of an object detection model.

It allows to:
* choose a model from the Tensorflow object detection model zoo
* prepare the data for augmentation and perform augmentation
* split the data into a training and validation
* convert this data into tf records (necessary for training)
* train/fine-tune the model


## Evaluating a model on a given set of test images

The EPRI_ObjectDetection_Evaluation.ipynb notebook allows for the evaluation of a given saved model.

It allows to:
* load a model
* extract the coco metrics from a given set of test images
* visualize custom metrics and confusion matrices
* perform visual evaluation on the images


## Authors

* Agnieszka Czeszumska (agaczesz@gmail.com)
* Frederic Colomer Martinez (frederic.colomer@gmail.com)
* Jenny Shih (jennywho86@gmail.com)
* Lennart Schmidt (lennartschmidt90@gmail.com)
* Sergey Komarov (skomarov1000@gmail.com)

### Installing development requirements
------------

    pip install -r requirements.txt

Note that errors might arise if the version of numpy is higher than 1.17.4.
