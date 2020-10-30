# -*- coding: utf-8 -*-
# Created by xieenning at 2020/10/27
import os
import sys

modules_path = os.path.join(os.getcwd(), '../')
sys.path.append(modules_path)

from argparse import Namespace, ArgumentParser
from datetime import datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.model import JointBERT
from src.data_module import JointBERTDataModule


def main(hparams: Namespace) -> None:
    """
    main training routine specific for this project.
    :param hparams:
    :return:
    """
    # step 0: seed everything for reproducibility.
    seed_everything(hparams.seed)

    # step 1: init `Model` and `DataModule`.
    data_module = JointBERTDataModule(hparams)
    data_module.prepare_data()
    # Add `slot_num_labels` and `intent_num_labels` to `hparams`.
    hparams.slot_num_labels = data_module.data_processor.slot_label_encoder.vocab_size
    hparams.intent_num_labels = data_module.data_processor.intent_label_encoder.vocab_size
    intent_weight = data_module.intent_weight

    model = JointBERT(hparams, intent_weight)

    # step 2: init callbacks for training.
    # Early stopping Callback.
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode
    )

    # Tensorboard Callback.
    tb_logger = TensorBoardLogger(
        save_dir='../saved_models/',
        version='version_' + datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
        name=''
    )

    # Model Checkpoint Callback.
    ckpt_path = os.path.join("../saved_models/", str(tb_logger.version), 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
        save_weights_only=True
    )

    # step 3: init trainer.
    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        # callbacks=[early_stop_callback],
        gradient_clip_val=1.0,
        gpus=hparams.gpus,
        log_gpu_memory='all',
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        precision=16,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
    )

    # step 4: start training
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = ArgumentParser(
        description="Joint BERT",
        add_help=True
    )
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved."
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="intent_f1_val_epoch", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"]
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help="Number of epochs with no improvement after which training will be stopped."
    )
    # epochs
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs."
    )
    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
        help="Limits training to a max number of epochs."
    )
    # Batching
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="Accumulated gradients run K small batches of size N before doing a backwards pass."
    )
    # gpu args
    parser.add_argument(
        "--gpus",
        default='0',
        type=str,
        help="Batch size to be used."
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help="If you don't want to use the entire dev set (for debugging or "
             "if it's huge), set how much of the dev set you want to use with this flag."
    )
    # each `LightningModule` defines arguments relevant to it
    parser = JointBERT.add_model_specific_args(parser)
    parser = JointBERTDataModule.add_data_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
