# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "../data/interim/Food-5K"


# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
#BASE_PATH = "kente_dataset"
BASE_PATH = "../../kente-cloth-authentication/data/processed"

# define the names of the training, testing, and validation
# directories
# TRAIN = "training"
TRAIN = "oc_training"
TEST = "evaluation"
VAL = "validation"

LABEL_INDEX = 0

# initialize the list of class label names
# CLASSES = ["non_food", "food"]
CLASSES = ["fake", "real"]

EXTRACT_FEATURES_TO_NPY=True
EXTRACT_FEATURES_TO_CSV=False

EXTRACT_AS_HSV=False
EXTRACT_IMAGE_NET_FEATURES=False
EXTRACT_MOBILENET_FEATURES=True

# set the batch size
BATCH_SIZE = 32

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.kente.cpickle"])
BASE_CSV_PATH = "output"

# Set the model to use
MODEL = "ONECLASS"
#MODEL = "SGD"

PROPORTION_TRAIN_CASES = 0.2