# -*- coding: utf-8 -*-
import click
import os
import logging
import numerox as nx
from pathlib import Path
from functools import reduce
import pandas as pd
import numpy as np
from tsfresh import extract_features
from dotenv import find_dotenv, load_dotenv

#  Assumes only one instance running
logger = logging.getLogger(__name__)

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
def convert_image_to_features(overwrite_any_existing, input_filepath, output_filepath):
    """
    """
    # logger.info('... converting filling in data if required...')

    ok_to_download = False

    if not input_filepath:
        input_filepath = '../../kente-cloth-authentication/data/'
    input_filepath = Path(input_filepath)

    if not output_filepath:
        output_filepath = '../data/processed/'
    output_filepath = Path(output_filepath)

    feature_filename = input_filepath/\
        Path(
            os.environ.get('FEATURE_FILENAME',
                           'features.npy')
        )


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
