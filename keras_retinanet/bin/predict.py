import sys
import os

import numpy as np
import argparse

import keras

# tf version 1.15.0-rc3
import tensorflow as tf

import cv2
import tifffile as tiff

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from .. import models
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.image import read_image, to_bgr, preprocess_image, resize_image

TRAINING_MIN_SIZE = 800
TRAINING_MAX_SIZE = 1333

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

def draw_detections(image, boxes, scores, labels, color=(255, 0, 0), label_to_name=None, score_threshold=0.05):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    # selection = np.where(scores > score_threshold)[0]

    # debug
    selection = np.where(scores > 0)[0]

    for i in selection:
        draw_box(image, boxes[i, :], color=color)

class RetinaNetWrapper(object):
    """docstring for RetinaNetWrapper"""
    def __init__(self, 
                model_path, 
                convert_model, 
                backbone,
                anchor_params  = None, 
                score_threshold=0.05,
                max_detections =2000,
                image_min_side =800,
                image_max_side =1333):
        super(RetinaNetWrapper, self).__init__()
        
        # load the model
        print('Loading model, this may take a second...')
        self.model = models.load_model(model_path, backbone_name=backbone)
        # optionally convert the model
        if convert_model:
            self.model = models.convert_model(self.model, anchor_params=anchor_params)

        tf.summary(self.model)

        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.image_min_side  = image_min_side
        self.image_max_side  = image_max_side

    def predict_large_image(self, image_path, save_path=None, image_type="planet"):
        tilesize_row = 1024
        tilesize_col = 1024

        # dataset     = gdal.Open(image_path, GA_ReadOnly)
        # size_column = dataset.RasterXSize
        # size_row    = dataset.RasterYSize

        image       = read_image(image_path)
        size_row    = image.shape[0]
        size_column = image.shape[1]

        for i in range(0, size_row, tilesize_row):
            for j in range(0, size_column, tilesize_col):
                rows = tilesize_row if i + tilesize_row < size_row else size_row - i
                cols = tilesize_col if j + tilesize_col < size_column else size_column - j
            
                # raw_image   = dataset.GetRasterBand(1).ReadAsArray(j, i, cols, rows).astype(np.float)      
                raw_image   = image[i: i + rows, j: j + cols, ...]
                

                image_bgr   = to_bgr(raw_image.copy())

                raw_image           = preprocess_image(raw_image, image_type)
                raw_image, scale    = resize_image(raw_image, min_side=self.image_min_side, max_side=self.image_max_side)

                if keras.backend.image_data_format() == 'channels_first':
                    raw_image = raw_image.transpose((2, 0, 1))

                print(raw_image.shape)

                # run network
                boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(raw_image, axis=0))[:3]
                # correct boxes for image scale
                boxes /= scale

                # select indices which have a score above the threshold
                indices = np.where(scores[0, :] > self.score_threshold)[0]

                # select those scores
                scores = scores[0][indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:self.max_detections]

                # select detections
                image_boxes      = boxes[0, indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[0, indices[scores_sort]]

                if save_path is not None:
                    draw_detections(image_bgr, image_boxes, image_scores, image_labels, score_threshold=self.score_threshold)
                    cv2.imwrite(os.path.join(save_path, '%d_%d.png' % (i, j)), image_bgr)
                
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)                
                print(type(image_detections))

                # copy detections to all_detections
                # all_detections = image_detections[image_detections[:, -1] == 0, :-1]

        # return all_detections

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('--image-path',       help='Path for image need detections.')
    parser.add_argument('--image-type',       help='Target image type. planet or terrasar. Default: planet', default="planet")
    parser.add_argument('--model',            help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    model = RetinaNetWrapper(args.model, args.convert_model, args.backbone,
                                anchor_params   = anchor_params,
                                score_threshold = args.score_threshold,
                                max_detections  = args.max_detections,
                                image_min_side  = args.image_min_side,
                                image_max_side  = args.image_max_side)

    model.predict_large_image(args.image_path, args.save_path, args.image_type)

    # import csv
    # with open(os.path.join(args.save_path, 'detections.csv'), mode='w') as csv_file:
    #     writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    #     for detection in all_detections:
    #         writer.writerow(detection)

if __name__ == '__main__':
    main()