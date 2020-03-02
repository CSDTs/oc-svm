# USAGE
# python extract_features.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from pyimagesearch import config
#from imutils import paths
from pathlib import Path
import numpy as np
import zarr
import pickle
import random
import logging
import os
import pdb

logger = logging.getLogger(__name__)

# load the ResNet50 network and initialize the label encoder
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)
le = None

# loop over the data splits
for split in (config.TEST, config.VAL, config.TRAIN):
	# grab all image paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([config.BASE_PATH, split])
	#imagePaths = list(paths.list_images(p))
	#imagePaths = list(Path(p).glob("*/*.jpg")) # for 5k
	imagePaths = list(Path(p).glob("*.jpg")) # for Kente

	print("[INFO] ... number of images in path {} ...".format(len(imagePaths)))
	# randomly shuffle the image paths and then extract the class
	print("[INFO] ... path is {} ...".format(p))

	# labels from the file paths
	random.shuffle(imagePaths)
	#labels = [str(p).split(os.path.sep)[-2] for p in imagePaths]
	#  for Kente we need to do things slightly differently
	labels = [p.name.split('_',1)[0] for p in imagePaths]


	# if the label encoder is None, create it
	if le is None:
		le = LabelEncoder()
		# le.fit(labels)
		#  the above assumes all label types are present but
		# in training they aren't
		le.fit(['fake','real'])

	# open the output CSV file for writing
	csvPath = os.path.sep.join([config.BASE_CSV_PATH,
		"{}.csv".format(split)])
	npPath = Path(
		os.path.sep.join([config.BASE_CSV_PATH,
		"{}.npy".format(split)])
		)
	if config.EXTRACT_FEATURES_TO_CSV:
		csv = open(csvPath, "w")
	if config.EXTRACT_FEATURES_TO_NPY:
		#  if already exists we wipe
		if npPath.exists():
			npPath.unlink()
			print("[INFO] ... deleted old extracted data")
		#npy = open(str(npPath), "ab")
		npy = zarr.open(str(npPath), mode="w")

	# loop over the images in batches
	for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		print("[INFO] processing batch {}/{}".format(b + 1,
			int(np.ceil(len(imagePaths) / float(config.BATCH_SIZE)))))
		print("[INFO] label encoding from path ...")
		batchPaths = imagePaths[i: i + config.BATCH_SIZE]
		batchLabels = le.transform(labels[i: i + config.BATCH_SIZE])
		batchImages = []
		print("[INFO] ... label encoded!")

		# loop over the images and labels in the current batch
		for imagePath in batchPaths:
			# load the input image using the Keras helper utility
			# while ensuring the image is resized to 224x224 pixels
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)

			# preprocess the image by (1) expanding the dimensions and
			# (2) subtracting the mean RGB pixel intensity from the
			# ImageNet dataset
			image = np.expand_dims(image, axis=0)
			image = imagenet_utils.preprocess_input(image)

			# add the image to the batch
			batchImages.append(image)

		# pass the images through the network and use the outputs as
		# our actual features, then reshape the features into a
		# flattened volume
		print("[INFO] generating features ... ")
		batchImages = np.vstack(batchImages)
		features = model.predict(batchImages, batch_size=config.BATCH_SIZE)
		features = features.reshape((features.shape[0], 7 * 7 * 2048))
		print("[INFO] ... generated features")

		if config.EXTRACT_FEATURES_TO_NPY:
			# add vector to feature file for faster read
			print("[INFO] extracting features to NPY ...")
			labels_by_column =\
				np.array(
					[0 if label == 'fake' else 1 for 
						label in labels[i: i + config.BATCH_SIZE]],
					ndmin= 2
				)
			npy.save(
				np.concatenate(
					(labels_by_column.T,
					  features),
					axis=1)
			)
			print("[INFO] ... saved features!")

		if config.EXTRACT_FEATURES_TO_CSV:
			# loop over the class labels and extracted features
			for (label, vec) in zip(batchLabels, features):
				# construct a row that exists of the class label and
				# extracted features
				vec = ",".join([str(v) for v in vec])
				csv.write("{},{}\n".format(label, vec))

	# close the CSV, numpy file
	if config.EXTRACT_FEATURES_TO_CSV:	
		csv.close()

# serialize the label encoder to disk
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()