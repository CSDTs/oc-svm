# USAGE
# python train.py

# import the necessary packages
from functools import partial
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_auc_score
from pyimagesearch import config
import pandas as pd
import numpy as np
import pickle
import os

# Packages for gridsearch, examining results
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  # for visualization
import seaborn as sns  # for pretty plot
sns.set_style("white")


def get_visualization_pipeline():
	pipeline =\
		Pipeline([('standard_scaler', StandardScaler()),
			 	  ('pca', PCA(n_components=3, random_state=42))]
		)
	return pipeline

def visualize_data(X, y, pred_y=None):
	my_dpi=96
	plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_zlabel("x_composite_3")

	sc =\
		ax.scatter(X[:, 0],
				   X[:, 1],
				   X[:, 2], 
				   c=y,  # color by outlier or inlier
				   cmap="Paired",
				   s=20,
				   alpha=0.75,
				   )

	# Plot x's for the ground truth outliers
	ax.scatter(X[y==-1, 0],
			   X[y==-1, 1],
			   zs=X[y==-1, 2], 
           	   lw=2,
			   s=60,
			   marker="x",
			   c="red")

	labels = np.unique(y)
	print("labels are: ", labels)
	handles = [plt.Line2D([],[], marker="o", ls="", 
						color=sc.cmap(sc.norm(yi))) for yi in labels]
	plt.legend(handles, labels)

	if pred_y is not None:
		print("[INFO] ... plotting predicted inliers")
		# Plot circles around the predicted outliers
		ax.scatter(X[pred_y == -1, 0],
				   X[pred_y  == -1, 1],
				   zs=X[pred_y== -1, 2], 
				   lw=4,
				   marker="o",
				   facecolors=None,
				   s=80)

	# make simple, bare axis lines through space:
	xAxisLine = ((min(X[:, 0]), max(X[:, 0])), (0, 0), (0,0))
	ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
	yAxisLine = ((0, 0), (min(X[:, 1]), max(X[:, 1])), (0,0))
	ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
	zAxisLine = ((0, 0), (0,0), (min(X[:, 2]), max(X[:, 2])))
	ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
	
	# label the axes
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	ax.set_title("Kente Cloth Inliers and Outliers")

	# seperately plot the pair wise histogram
	sns.pairplot(
		pd.DataFrame({
			"x":X[:,0],
			"y":X[:,1],
			"z":X[:,2], 
			"label":y}),
		hue="label",
		corner=True,
		diag_kind='kde'
	)  # todo: set same colors as prior plot


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

if config.MODEL == 'SGD':
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

if config.MODEL == "ONECLASS":
	#see: https://hackernoon.com/one-class-classification-for-images-with-deep-features-be890c43455d
	#see: https://sdsawtelle.github.io/blog/output/week9-anomaly-andrew-ng-machine-learning-with-python.htmlfrom sklearn.model_selection import StratifiedKFold
	from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
	from sklearn.model_selection import GridSearchCV
	from sklearn.metrics import f1_score, recall_score, make_scorer
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA, KernelPCA
	from sklearn.ensemble import IsolationForest
	from sklearn.svm import OneClassSVM
	from sklearn.neighbors import LocalOutlierFactor, NeighborhoodComponentsAnalysis

	def load_data(data_set,
				  base_path=config.BASE_CSV_PATH,
				  remap_y_values={0:-1},
				  use_hsv=True,
				  subset=True):
		hsv = ''
		if use_hsv:
			hsv = 'hsv.'
		data = np.load(
			os.path.sep.join(
				[base_path,
				 f"{data_set}.{hsv}npy"]
			)
		)

		X, y =\
			data[:, config.LABEL_INDEX+1:],\
				data[:, config.LABEL_INDEX]

		if remap_y_values:
			y =\
				np.array(
					[remap_y_values.get(value, value) for value in y]
			)

		if subset:
			X, y = subset_data(X, y)

		return X, y

	def subset_data(X, y, fraction=config.PROPORTION_TRAIN_CASES):
		indices = np.random\
					.randint(X.shape[0],
							 size=int(X.shape[0]*fraction))
		return X[indices], y[indices]

	print("[INFO] Loading train, validation and test into memory ...")
	# get all of train, evaluation generator
	# Note we use two class (insead of one class), for NCA
	#  Some of the outliers got moved to validation so, the validation
	# set is slightly baised. The test set should have only outliers not seen before
	#
	# So we do hyperparameter searching, still, on the training, validation set
	# with validation hold out to double check. Then we'll look at test wiht the best one
	# hopefullyit's all good.
	X_train, y_train = load_data(data_set="oc_training.mobile",
							     use_hsv=False,
								 subset=False)  # note one class, need to steal from val

	X_val, y_val = load_data(data_set="validation.mobile",
							 use_hsv=False,
							 subset=False)

	X_test, y_test = load_data(data_set="evaluation.mobile",
							   use_hsv=False,
							   subset=False)
	print("[INFO] ... loaded into memory")

	# ... X_train was designed to be one class only because I thought I was going to do
	# an SVM approach but that turned out to be too big of an assumption.
	# so, now need to mix X_val and X_train to get a balance of counterfeit and authentic examples

	X_train_and_val = np.row_stack((X_train, X_val))
	y_train_and_val = np.hstack((y_train, y_val))
	stratified_splitter =\
		StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
	new_train_index, new_test_index =\
		next(stratified_splitter.split(X_train_and_val, y_train_and_val))
	X_train, y_train =\
		X_train_and_val[new_train_index],
		y_train_and_val[new_train_index]
	X_val, y_val =\
		X_train_and_val[new_test_index],
		y_train_and_val[new_test_index]

	print("[INFO] Applying standard scaling ...")
	ss = StandardScaler()
	ss.fit(X_train)
	X_train = ss.transform(X_train)
	X_val = ss.transform(X_val)
	X_test = ss.transform(X_test)
	print("[INFO] ... scaled")

	#  Here we apply a variety of dimensionality reduction techniques,
	# to be evaluated on held out validation data and within parameter search
	reduction_sizes =\
		{"reduction_125": int(X_train.shape[0]*0.125),}
		 "reduction_50": int(X_train.shape[0]*0.5), 
		 "reduction_25": int(X_train.shape[0]*0.25)}
	dimensionality_reducers =\
		{"PCA": PCA,
		 "KPCA": KernelPCA,
		 "NCA": NeighborhoodComponentsAnalysis}

	kpca_args =\
		{"n_jobs": -1,
		 "kernel": "rbf",
		 "copy_X": False}

	fast_ica_args =\
		{"n_jobs": -1,
		 "algorithm": "parallel"}

	the_reducers = []

	for size in reduction_sizes.values():
		for name, reducer in dimensionality_reducers.items():
			print(f"[INFO] reducing with {name} on size {size} ...")
			args = {"n_components": size,
					"random_state": 42}
			if name in ["KPCA"]:
				args.update(kpca_args)
			if name in ["NCA"]:
				the_reducers.append(
					(name, reducer(**args).fit(X_train, y_train))
				)
			else:
				the_reducers.append(
					(name, reducer(**args).fit(X_train))
				)
			print(f"[INFO] ... finished with instance of {name}")

	#  ... and also do fast ICA with 4 and 2 sources
	ica_reducers =\
		{"FastICA": FastICA}
	for size in [2, 4]:
		for name, reducer in ica_reducers:
			print(f"[INFO] ... reducing with {name} on size {size}")
			args = {"n_components": size,
					"random_state": 42}
			the_reducers.append(
				(name, reducer(**args).fit(X_train))
			)

	# # PCA it (that's what they did)
	# print("[INFO] Applying PCA ...")
	# kpca = KernelPCA(n_jobs=-1, n_components=256) #X_train is 505?
	# kpca = kpca.fit(X_train)

	#  DO NCA instead
	#   Note that this is y aware, so the parameter eval should 

	# thebelow is taking too long, maybe just switch to PCA
	#----
	nca1 = NeighborhoodComponentsAnalysis(n_components=int(1280*7*7*0.5), verbose=10)
	nca1.fit(X_train, y_train)

	nca2 = NeighborhoodComponentsAnalysis(n_components=int(1280*7*7*0.25), verbose=10)
	nca2.fit(X_train, y_train)

	nca3 = NeighborhoodComponentsAnalysis(n_components=int(1280*7*7*0.10, verbose=10)
	nca3.fit(X_train, y_train)

	nca4 = NeighborhoodComponentsAnalysis(n_components=int(1280*7*7*0.05), verbose=10)
	nca4.fit(X_train, y_train)

	nca4 = NeighborhoodComponentsAnalysis(n_components=3, verbose=10)
	nca4.fit(X_train, y_train)
	# -------
	#  from this we take n_components 100 to be right at the knee
	pca1 = PCA(n_components=int(X_train.shape[0]*0.50), whiten=True).fit(X_train)
	print('Explained variance percentage = %0.2f' % sum(pca1.explained_variance_ratio_))

	pca2 = PCA(n_components=int(X_train.shape[0]*0.25), whiten=True).fit(X_train)
	print('Explained variance percentage = %0.2f' % sum(pca2.explained_variance_ratio_))

	pca3 = PCA(n_components=int(X_train.shape[0]*0.125), whiten=True).fit(X_train)
	print('Explained variance percentage = %0.2f' % sum(pca3.explained_variance_ratio_))

	pca4 = PCA(n_components=int(X_train.shape[0]*0.05), whiten=True).fit(X_train)
	print('Explained variance percentage = %0.2f' % sum(pca4.explained_variance_ratio_))


	# plot first two dimensions
	from matplotlib import cm
	X_embedded = nca.transform(X_train)

	# basically a huge blob of outliers and inliers spread out in a v shape
	visualize_data(X_embedded[:,:4], y_train)

	# Use a scree plot to pick right dimensions


	X_train_k = kpca.transform(X_train)
	X_val_k = kpca.transform(X_val)

	X_train = pca.transform(X_train)
	X_val = pca.transform(X_val)

	#X_test = pca.transform(X_test)
	print(f"[INFO] ... transformed train shape: {X_train.shape}, val shape: {X_val.shape}")

	print("[INFO] Entering hyper parameter search for IsolationForest using validation data for F1 ...")
	val_cuttoff = 100 # we use the rest of validate to test out of sample

	X_train_and_val = np.row_stack((X_train_k, 
								    X_val_k[:val_cuttoff,:]))
	y_train_and_val = np.hstack((y_train, 
								 y_val[:val_cuttoff])) # truncated at val_cuttoff

	f1_scorer = make_scorer(f1_score)
	# I think 'micro' implements recomendation of Aples to Apples in CV studies

	contamination = {"contamination": list(np.linspace(0, 0.10, 2))+['auto']}
	number_estimators = {"n_estimators": np.linspace(3, 20, num=3, dtype=int)}
	max_features = {"max_features": np.linspace(0.1, 1.0, 3)}
	kernel = {"kernel":["linear", "rbf", "poly", "sigmoid"]}
	degree  = {"degree": np.linspace(2, 10, num=3, dtype=int)}
	nu = {"nu": np.linspace(0.01, 1.0, 4)}
	gamma = {"gamma":['auto', 1/512.0]}  #
	n_neighbors = {"n_neighbors": np.linspace(1, 40, num=10, dtype=int)}
	metric = {"metric":\
		['cityblock', 'cosine', 'euclidean',
		 'l1', 'l2', 'manhattan'] +\
		['braycurtis', 'canberra', 'chebyshev',
		 'correlation', 'dice', 'hamming', 
		 'jaccard', 'kulsinski', 'mahalanobis',
		 'minkowski', 'rogerstanimoto', 'russellrao',
		 'seuclidean', 'sokalmichener', 'sokalsneath',
		 'sqeuclidean', 'yule']}
	novelty = {"novelty":[True]}

	parameter_grid = {#**contamination,
					  #**number_estimators,
					  #**max_features,
					  "n_jobs":[-1],
					  ##**kernel,
					  ##**degree,
					  ##**nu,
					  ##**gamma}
					  ##"random_state":[42]}
					  **n_neighbors,
					  **metric,
					  **novelty}
 
	folds = StratifiedKFold(n_splits=3).split(X_train_and_val, y_train_and_val)
	search = GridSearchCV(
		estimator=LocalOutlierFactor(),
		param_grid=parameter_grid,
		scoring='balanced_accuracy',
		cv=folds,
		verbose=10,
		n_jobs=-1)
	search.fit(X_train_and_val, y_train_and_val)
	#search.fit(X_train_k, y_train)

	optimal_forest = search.best_estimator_

	X_val_test_with = X_train_and_val[val_cuttoff:]
	y_val_test_with = y_train_and_val[val_cuttoff:]  # truncated AFTER val_cuttoff

	preds = optimal_forest.predict(X_val_test_with)

	print(classification_report(y_val_test_with, preds))

	#  Test out of sample...
	X_val_plot_with = get_visualization_pipeline().fit_transform(
		X_val_test_with)
	visualize_data(X_val_test_with,
			       y_val_test_with,
				   preds)

	# ------
	# special loop for OC svm
	from sklearn.model_selection import ParameterGrid

	for a_set_of_parameters in ParameterGrid(parameter_grid):
		clf = OneClassSVM(**a_set_of_parameters)
		clf.fit(X_train_k)
		preds = optimal_forest.predict(X_val_test_with)
		print(classification_report(y_val_test_with, preds))
		print(a_set_of_parameters)
		print("\n")
	

	# ------


	#  plot fit results against 

	print("[INFO] training OC-SVM, Isolation Forest ...")
	# Train classifier and obtain predictions for OC-SVM
	oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
	if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search
	print("[INFO] ... trained")

	print("[INFO] fitting OC-SVM, Isolation Forest ...")
	oc_svm_clf.fit(X_train)
	if_clf.fit(X_train)
	print("[INFO] ... fitted")

	print("[INFO] predicting on validation data ...")
	oc_svm_preds = oc_svm_clf.predict(X_val)
	if_preds = if_clf.predict(X_val)
	print(f"[INFO] ... predicted oc_svm shape: {oc_svm_preds.shape}")

	print("[INFO] producing classification report ...")
	# prediction, with classification report
	print("One Class SVM...")
	print(f"[INFO] class prediction counts {np.unique(oc_svm_preds, return_counts=True)}")
	print(classification_report(y_val, oc_svm_preds,
		target_names=['fake', 'real']))
	print("ROC AUC Score: ", roc_auc_score(y_val, oc_svm_preds))

	print("Isolation Forest ...")
	print(f"[INFO] class prediction counts {np.unique(if_preds, return_counts=True)}")	
	print(classification_report(y_val, if_preds,
		target_names=['fake', 'real']))
	print("ROC AUC Score: ", roc_auc_score(y_val, if_preds))
	print("[INFO] done!")