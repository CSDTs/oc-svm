
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
    logger.info('... converting filling in data if required...')

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


    if numerai_data_exists and overwrite_any_existing:
        logger.info('... data exists ...')
        ok_to_download = True
    elif not numerai_data_exists:
        logger.info('... no data exists ...')
        ok_to_download = True
    elif not overwrite_any_existing and numerai_data_exists:
        logger.info('... data exists as .zip or as .hdf, skipping this step ...')

    if ok_to_download:
        logger.info(f'... downloading new data into {interim_filepath} ...')
        data = nx.download('numerai_dataset.zip', single_precision=True)
        data.save(
           str(interim_filepath)+'/'+hdf_numerai_data
        )

@main.command()
@click.option('--eras', '-e', required=True, multiple=True, type=int)
@click.option('--number_of_features', '-f', default=50, type=int)  # this really should be the whole thing
@click.option('--interim_filepath', '-i', required=False)
@click.option('--output_filepath', '-o', required=False)
def tsfeatures(eras, number_of_features, interim_filepath, output_filepath):
    interim_filepath = Path(
        os.environ.get('INTERIM_PATH',
                       './data/interim/')
    )
    hdf_numerai_data = interim_filepath/\
        Path(
            os.environ.get('HDF_DATASET_NAME',
                           'numerai_dataset.hdf')
        )

    if not output_filepath:
        output_filepath = './data/processed/'
    output_filepath = Path(output_filepath)

    ts_fresh_name = os.environ.get('TS_FRESH_NAME', 'ts_fresh_name.csv')

    logger.info(f'... doing featurization with {eras} ... in {interim_filepath} ...')

    #  https://github.com/numerai/numerox/blob/master/numerox/examples/data.rst 
    # claims this is a faster way to load data but it takes some time :( 35 seconds
    logger.info('\t... loading data ...')
    data = nx.load_data(str(hdf_numerai_data))
    logger.info('\t... loaded data!')

    logger.info('\t... subseting features from data frame ...')
    feature_offset = 2
    max_number_of_features = number_of_features
    eras = eras
    individual_era_masks =\
        (data.era_float == an_era for an_era in eras)
    era_mask = reduce(np.logical_or, individual_era_masks)

    df = data[era_mask].y\
                       .df\
                       .iloc[:,
                             [0]+list(  # for the era identifer
                                range(
                                    feature_offset,
                                    feature_offset+max_number_of_features)
                                )  #  takes first number features and era, as id column
                        ]
    y = pd.Series(eras)  # assignment each era to its era number, try to distinguish
    y.index+=1

    #  extract_relevant_features() applies Benjamini-Yekutieli procedure
    # to determine what features to drop. While this is nice I'm not
    # 100% convinced that it accounts for higher order interactions or
    # mutual entropy that could drive better predictions, esp. in such
    # a non-linear problem like this
    logger.info('\t... extracting TS fresh features (will take a looooong time)...')
    X_tsfresh = extract_features(df, column_id='era')  # uses all cores
    X_tsfresh.to_csv(str(output_filepath/ts_fresh_name))
    logger.info('\t... wrote TS fresh feature dataframe!')

main.add_command(makeinterim)
main.add_command(tsfeatures)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
