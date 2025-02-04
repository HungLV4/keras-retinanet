import sys
import os

import numpy as np
import argparse

import keras

# tf version 1.15.0-rc3
import tensorflow as tf

import cv2
import csv
import geoio

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from snappy import ProductIO, PixelPos, GeoPos

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from tqdm import tqdm

from .. import models
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.image import  read_image_bgr, to_bgr, preprocess_image, resize_image
from ..utils.geo import *

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
				image_max_side =1333
	):
		super(RetinaNetWrapper, self).__init__()

		# load the model
		print('Loading model, this may take a second...')
		with tf.device('/cpu:0'):
			self.model = models.load_model(model_path, backbone_name=backbone)

		# optionally convert the model
		if convert_model:
			self.model = models.convert_model(self.model, anchor_params=anchor_params)

		print(self.model.summary())

		self.score_threshold = score_threshold
		self.max_detections  = max_detections
		self.image_min_side  = image_min_side
		self.image_max_side  = image_max_side

	def predict(self, raw_image, image_type="planet"):
		image        = preprocess_image(raw_image.copy(), image_type=image_type)
		image, scale = resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)
		if keras.backend.image_data_format() == 'channels_first':
			image = image.transpose((2, 0, 1))

		# run network
		input_image = np.expand_dims(image, axis=0)

		boxes, scores, labels = self.model.predict_on_batch(input_image)[:3]
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

		return image_boxes, image_scores, image_labels

	def predict_large_image(self, image_path, resolution, vis_path=None, scale_factor=0.2, save_path=None, image_type="planet"):
		tilesize_row = 1025
		tilesize_col = 1025

		file_type   = os.path.basename(image_path).split(".")[-1]
		basename    = os.path.basename(image_path).split(".")[0]

		if file_type in ["tif", "TIF", "tiff", "TIFF"]:
			dataset     = gdal.Open(image_path, GA_ReadOnly)
			size_column = dataset.RasterXSize
			size_row    = dataset.RasterYSize
			size_band   = dataset.RasterCount

			xyToLatLonFunc  = xyToLatLonTiff
			readTileFunc    = readTiffTile
		elif file_type in ["dim", "DIM"]:
			dataset     = ProductIO.readProduct(image_path)
			size_column = dataset.getSceneRasterWidth()
			size_row    = dataset.getSceneRasterHeight()
			size_band   = len(dataset.getBandNames())

			xyToLatLonFunc  = xyToLatLonDim
			readTileFunc    = readDimTile
		else:
			print("File type %s not supported" % file_type)
			return

		# read rgb image for visualization
		if vis_path is None:
			image_bgr       = readTileFunc(dataset, 0, 0, size_column, size_row, size_band, scale_factor=scale_factor)
			if image_type == "terrasar":
				# TerraSAR image has only one channel
				# raw_image     = np.expand_dims(raw_image, axis=2)
				image_bgr     = np.repeat(image_bgr, 3, axis=2)
			elif image_type == "planet":
				reverse = False
				if image_bgr.shape[2] == 3:
					reverse = True
				image_bgr = image_bgr[..., :3]
				if reverse:
					image_bgr = image_bgr[..., ::-1].copy()
			image_bgr       = to_bgr(image_bgr)
		else:
			image_bgr 		= read_image_bgr(vis_path)

		all_detections  = np.array([[0, 0, size_column - 1, size_row - 1]])
		for i in tqdm(range(0, size_row, tilesize_row)):
			for j in tqdm(range(0, size_column, tilesize_col)):
				rows = tilesize_row if i + tilesize_row < size_row else size_row - i
				cols = tilesize_col if j + tilesize_col < size_column else size_column - j

				raw_image   = readTileFunc(dataset, j, i, cols, rows, size_band)
				if image_type == "terrasar":
					# TerraSAR image has only one channel
					# raw_image     = np.expand_dims(raw_image, axis=2)
					raw_image     = np.repeat(raw_image, 3, axis=2)
				elif image_type == "planet":
					reverse = False
					if raw_image.shape[2] == 3:
						reverse = True
					raw_image = raw_image[..., :3]
					if reverse:
						raw_image = raw_image[..., ::-1].copy()

				image_boxes, image_scores, image_labels  = self.predict(raw_image, image_type=image_type)
				# add offset to image_boxes
				image_boxes[..., 0] += j
				image_boxes[..., 1] += i
				image_boxes[..., 2] += j
				image_boxes[..., 3] += i

				# concatenate results
				all_detections = np.concatenate([all_detections, image_boxes], axis=0)

				if save_path is not None:
					resize_image_boxes = image_boxes * scale_factor
					draw_detections(image_bgr, resize_image_boxes, image_scores, image_labels, score_threshold=self.score_threshold)

		with open(os.path.join(save_path, '%s.csv' % basename), mode='w') as csv_file:
			writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			# write down detections
			# the first line will be extent of image
			for i, d in enumerate(all_detections):
				if i == 0:
					ulx, uly = xyToLatLonFunc(dataset, d[0], d[1])
					brx, bry = xyToLatLonFunc(dataset, d[2], d[3])
					if ulx > 90 or ulx < -90 or uly > 180 or uly < -180:
						ulx, uly = utmToLatLng(48, ulx, uly)
						brx, bry = utmToLatLng(48, brx, bry)
					writer.writerow([ulx, uly, brx, bry])
				else:
					lx, ly = xyToLatLonFunc(dataset, (d[0] + d[2]) / 2, (d[1] + d[3]) / 2)
					if lx > 90 or lx < -90 or ly > 180 or ly < -180:
						lx, ly = utmToLatLng(48, lx, ly)

					writer.writerow([lx, ly, (d[2] - d[0]) * resolution, (d[3] - d[1]) * resolution])

		cv2.imwrite(os.path.join(save_path, '%s_vis.png' % basename), image_bgr)

def parse_args(args):
	""" Parse the arguments.
	"""
	parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
	parser.add_argument('--image-path',       help='Path for image need detections.')
	parser.add_argument('--vis-path',         help='Path for visualize image.', default=None)
	parser.add_argument('--vis-scale-factor', help='Scale factor for visualize image.', type=float, default=0.2)
	parser.add_argument('--res', 			  help='Image resolution.', type=float, default=2.5)
	parser.add_argument('--image-type',       help='Target image type. planet or terrasar. Default: planet', default="planet")
	parser.add_argument('--model',            help='Path to RetinaNet model.')
	parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
	parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
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

	model.predict_large_image(args.image_path, args.res, args.vis_path, args.vis_scale_factor, args.save_path, args.image_type)

if __name__ == '__main__':
	main()