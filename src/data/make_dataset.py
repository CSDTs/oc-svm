# -*- coding: utf-8 -*-
import click
import os
import logging
from pathlib import Path
from functools import reduce
import requests
import shutil
from dotenv import find_dotenv, load_dotenv

#  Assumes only one instance running
logger = logging.getLogger(__name__)

# see: https://stackoverflow.com/a/39217788/3662899
def download_file(url, file_path, total_size=447*1024):
    with click.progressbar(length=total_size, label='Downloading files') as bar:
        local_filename = file_path
        with requests.get(url, stream=True, timeout=2) as r:
            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f, length=16*1024*1024)
                bar.update(r.raw.length)

    return local_filename

@click.group()
def main():
    """
    Helps generate data set, note test dataset (from hackernoon) that
    excercises this pipeline
    """
    pass

@main.command()
@click.option('--overwrite_any_existing', default=False, type=bool)
@click.option('--input_filepath', '-i', required=False)
@click.option('--output_filepath', '-o', required=False)
def food5k(overwrite_any_existing, input_filepath, output_filepath):
    """
    """
    logger.info('... converting filling in data if required...')

    ok_to_download = overwrite_any_existing

    if not input_filepath:
        input_filepath = './data/raw/'
    input_filepath = Path(input_filepath)

    if not output_filepath:
        output_filepath = './data/interim/'
    output_filepath = Path(output_filepath)

    food5k_path = input_filepath/Path('Food-5k.zip')

    if (not food5k_path.exists()) or (food5k_path.exists() and ok_to_download):
        #  we pull down the 5k data
        FOOD_5K_FILE = os.environ.get('FOOD_5K_URL',
                                     'http://grebvm2.epfl.ch/lin/food/Food-5K.zip')
        try:
            requests.get(FOOD_5K_URL, str(food5k_path))
        except requests.Timeout:
            logger.info('Server timeout, unable to d/l the file.\nYou must download directly from https://www.kaggle.com/binhminhs10/food5k/data')

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
