import json

import tensorflow as tf


def get_default_hparams():
    # yapf: disable
    hparams = tf.contrib.training.HParams(
        debug_size=100000000,
        val_debug_size=10000000,
        batch_size=100,
        shuffle=True,
        dataset_type="images",
        image_type="grayscale",
        load_filepath="",
        model_type="multiclass",
        recurrent_dropout_prob=0.0,
        opt_type="sgd",
        opt_lr=0.01,
        opt_momentum=0.9,
        opt_weight_decay=1e-4,
        grad_clip=1.0,
        grad_norm=1000.0,
        lr_cycle_ratio=0.1,
        lr_cycle_decay=0.99,
        early_stopping_patience=50,
        max_epochs=500
    )
    # yapf: enable
    return hparams


def merge_args_hparams(args):
    hps = get_default_hparams()
    if args.hparams is not None:
        hps.parse(args.hparams)
    args.__dict__.update(dict(hps.values()))
    return args
