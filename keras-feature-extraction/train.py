# USAGE
# python train.py

# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from pyimagesearch import config
import pandas as pd
import numpy as np
import pickle
import os

def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
	# open the input file for reading
	f = open(inputPath, "r")

	# loop indefinitely
	while True:
		# initialize our batch of data and labels
		data = []
		labels = []

		# keep looping until we reach our batch size
		while len(data) < bs:
			# attempt to read the next row of the CSV file
			row = f.readline()

			# check to see if the row is empty, indicating we have
			# reached the end of the file
			if row == "":
				# reset the file pointer to the beginning of the file
				# and re-read the row
				f.seek(0)
				row = f.readline()

				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break

			# extract the class label and features from the row
			row = row.strip().split(",")
			label = row[0]
			label = to_categorical(label, num_classes=numClasses)
			features = np.array(row[1:], dtype="float")

			# update the data and label lists
			data.append(features)
			labels.append(label)

		# yield the batch to the calling function
		yield (np.array(data), np.array(labels))

# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# derive the paths to the training, validation, and testing CSV files
trainPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TRAIN)])
valPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.VAL)])
testPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TEST)])

# determine the total number of images in the training and validation
# sets
# totalTrain = sum([1 for l in open(trainPath)])
# totalVal = sum([1 for l in open(valPath)])
totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])


# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

print("[INFO] setting up generators for train, test and validation  ...")
print(f"[INFO] ... train from {trainPath}")
# construct the training, validation, and testing generators
trainGen = csv_feature_generator(trainPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="train")
valGen = csv_feature_generator(valPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="eval")
testGen = csv_feature_generator(testPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="eval")
print("[INFO] ... setup train, test and validation generators!")


if config.MODEL == 'SGD':
	number_of_epochs = 3
	# define our simple neural network
	model = Sequential()
	model.add(Dense(256, input_shape=(7 * 7 * 2048,), activation="relu"))
	model.add(Dense(16, activation="relu"))
	model.add(Dense(len(config.CLASSES), activation="softmax"))

	# compile the model
	opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	print("[INFO] training simple network...")
	H = model.fit_generator(
		trainGen,
		steps_per_epoch=totalTrain // config.BATCH_SIZE,
		validation_data=valGen,
		validation_steps=totalVal // config.BATCH_SIZE,
		epochs=number_of_epochs)

	# make predictions on the testing images, finding the index of the
	# label with the corresponding largest predicted probability, then
	# show a nicely formatted classification report
	print("[INFO] evaluating network...")
	predIdxs = model.predict_generator(testGen,
		steps=(totalTest //config.BATCH_SIZE) + 1)
	predIdxs = np.argmax(predIdxs, axis=1)
	print(classification_report(testLabels, predIdxs,
		target_names=le.classes_))

if config.MODEL == "one-class":
	#see: https://hackernoon.com/one-class-classification-for-images-with-deep-features-be890c43455d
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	from sklearn.ensemble import IsolationForest
	from sklearn import svm	

	# get all of train, evaluation generator
	X_train = np.concatenate(
		np.array(trainGen),
		np.array(valGen)
	)
	X_test = np.array(testGen)
	#  scale it (that's what they did)
	ss = StandardScaler()
	ss.fit(X_train)
	X_train = ss.transform(X_train)
	X_test = ss.transform(X_test)

	# PCA it (that's what they did)
	pca = PCA(n_components=512, whiten=True)
	pca = pca.fit(X_train)
	print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	# Train classifier and obtain predictions for OC-SVM
	oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
	if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

	oc_svm_clf.fit(X_train)
	if_clf.fit(X_train)

	oc_svm_preds = oc_svm_clf.predict(X_test)
	if_preds = if_clf.predict(X_test)	

	# prediction, with classification report
	print("One Class SVM...")
	print(classification_report(testLabels, oc_svm_preds,
		target_names=le.classes_))

	print("Isolation Forest ...")
	print(classification_report(testLabels, if_preds,
		target_names=le.classes_))
