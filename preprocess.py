from pathlib import Path

import click
import numpy as np
import pandas as pd

from utils import preprocess_data


def transform_and_save(source_path, target_path):
    # read data
    data = pd.read_csv(source_path)

    # transform
    data = preprocess_data(data)

    # output
    np.save(str(target_path.absolute()), data)


@click.command()
@click.option('-s', '--source', help='path to the dir containing train and test csv', required=True)
@click.option('-t', '--target', help='path to the dir for persisting transformed numpy array', required=True)
def main(source, target):
    source_dir_path = Path(source)
    train_data_path = source_dir_path / 'UNSW_NB15_training-set.csv'
    test_data_path = source_dir_path / 'UNSW_NB15_testing-set.csv'

    # prepare target path
    target_dir_path = Path(target)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_train_path = target_dir_path / 'train'
    target_test_path = target_dir_path / 'test'

    # processing
    transform_and_save(train_data_path, target_train_path)
    transform_and_save(test_data_path, target_test_path)


if __name__ == '__main__':
    main()
