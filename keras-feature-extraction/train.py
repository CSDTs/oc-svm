# USAGE
# python train.py

from pyimagesearch import config

def main():
    if config.MODEL == "ONECLASS":
        print("Running one class model")
        #see: https://hackernoon.com/one-class-classification-for-images-with-deep-features-be890c43455d
        #see: https://sdsawtelle.github.io/blog/output/week9-anomaly-andrew-ng-machine-learning-with-python.htmlfrom sklearn.model_selection import StratifiedKFold
        from one_class_model import run_one_class_model
        run_one_class_model()

    elif config.MODEL == 'SGD':
        from sgd_model import run_sgd_model
        run_sgd_model()


if __name__ == "__main__":
    main()
