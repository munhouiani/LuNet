from pathlib import Path

import click

from ml.utils import train_lunet_binary, train_lunet_multiclass


@click.command()
@click.option('-t', '--train_data_path', help='path to train set', required=True)
@click.option('-v', '--val_data_path', help='path to val set', required=True)
@click.option('-m', '--model_path', help='path to persist model', required=True)
@click.option('-a', '--task', help='task type (Option: "binary" or "multiclass")', required=True)
@click.option('--use_gpu', help='whether to use gpu', default=True)
def main(train_data_path, val_data_path, model_path, use_gpu, task):
    # prepare path for model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gpu:
        gpu = -1
    else:
        gpu = None

    if task == 'binary':
        train_lunet_binary(train_data_path=train_data_path, test_data_path=val_data_path, gpu=gpu,
                           model_path=model_path)
    elif task == 'multiclass':
        train_lunet_multiclass(train_data_path=train_data_path, test_data_path=val_data_path, gpu=gpu,
                               model_path=model_path)

    else:
        exit('Error task type')


if __name__ == '__main__':
    main()
