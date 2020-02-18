# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv
from gzip import GzipFile
import zipfile
from keras.applications.resnet50 import ResNet50

#  Assumes only one instance running
logger = logging.getLogger(__name__)

def extract_images(image_path, dest_path):
    path = Path(image_path)
    if path.name.endswith('.zip'):
        # see: https://www.peterbe.com/plog/fastest-way-to-unzip-a-zip-file-in-python
        with open(str(image_path), 'rb') as f:
            zf = zipfile.ZipFile(f)
            zf.extractall(dest_path)

#@click.command()
@click.group()
def main():
    """ Runs data processing scripts to turn data from ../data/processed
    into features
    """
    pass

@main.command()
@click.option('--overwrite_any_existing', default=False, type=bool)
@click.option('--input_filepath', '-i', required=False)
@click.option('--output_filepath', '-o', required=False)
def make_resnet50_features(overwrite_any_existing, input_filepath, output_filepath):
    """
    """
    logger.info('... reading in files ...')

    ok_to_download = overwrite_any_existing

    if not input_filepath:
        # raw image data
        input_filepath = '../../kente-cloth-authentication/data/images.npy'
    input_filepath = Path(input_filepath)

    if not output_filepath:
        output_filepath = './data/processed/'
    output_filepath = Path(output_filepath)

    interim_filepath = Path('./data/interim')

    feature_filename = input_filepath/\
        Path(
            os.environ.get('FEATURE_FILENAME',
                           'resnet_50_features.npy')
        )

    image_data = None
    if not feature_filename.name.endswith('.npy'):
        logger.info(f'... unpacking {} ...'.format(str(feature_filename)))
        interim_file_path = \
            interim_filepath / Path(input_filepath.name)
        extract_images(str(input_filepath),
                       str(interim_file_path))
        #  now we hava  bunch of images (I'll assume .jpg/.jpeg) in 
        # interim_file_path. As a kludge, we only read 1 direct deep
        # since I know the 5K image set (a test data set) has a weird
        # duplication issue

    else:
        image_data = read_image(str(input_filepath))



    image_h = image_data[0].shape[0]
    image_w = image_data[0].shape[1]
    image_depth = image_data[0].shape[2]

    logger.info('... constructing ResNet50 CNN ...')
    resnet_model = \
        ResNet50(input_shape=(image_h,
                              image_w,
                              image_depth),
                 weights='imagenet',
                 include_top=False)
    logger.info('... constructing features array ...')
    features_array = \
        resnet_model.predict(image_depth)
    #  write features_array to disk (processed)

main.add_command(convert_image_to_features)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
