from argparse import Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ml.model import LuNet


def train_lunet(input_dim, c1_out, c1_kernel, c2_out, c2_kernel, c3_out, c3_kernel, final_conv, final_kernel, out_dim,
                train_data_path, test_data_path, num_epoch, gpu, model_path, logger):
    hparams = Namespace(**{
        'input_dim': input_dim,
        'c1_out': c1_out,
        'c1_kernel': c1_kernel,
        'c2_out': c2_out,
        'c2_kernel': c2_kernel,
        'c3_out': c3_out,
        'c3_kernel': c3_kernel,
        'final_conv': final_conv,
        'final_kernel': final_kernel,
        'out': out_dim,
        'train_data_path': train_data_path,
        'val_data_path': test_data_path,
    })
    model = LuNet(hparams)
    trainer = Trainer(max_epochs=num_epoch, gpus=gpu, logger=logger)
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def train_lunet_binary(train_data_path, test_data_path, gpu, model_path):
    # prepare logger
    logger = TensorBoardLogger('lunet_binary_model_logs', 'lunet_binary_model')

    train_lunet(
        input_dim=196, c1_out=64, c1_kernel=64, c2_out=128, c2_kernel=128, c3_out=256, c3_kernel=256, final_conv=512,
        final_kernel=512, out_dim=2, train_data_path=train_data_path, test_data_path=test_data_path, num_epoch=10,
        gpu=gpu, model_path=model_path, logger=logger
    )


def train_lunet_multiclass(train_data_path, test_data_path, gpu, model_path):
    # prepare logger
    logger = TensorBoardLogger('lunet_multiclass_model_logs', 'lunet_multiclass_model')

    train_lunet(
        input_dim=196, c1_out=64, c1_kernel=64, c2_out=128, c2_kernel=128, c3_out=256, c3_kernel=256, final_conv=512,
        final_kernel=512, out_dim=10, train_data_path=train_data_path, test_data_path=test_data_path, num_epoch=10,
        gpu=gpu, model_path=model_path, logger=logger
    )
