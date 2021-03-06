import argparse
import os

import ignite
from ignite.engine import Events
import numpy as np
import torch.utils.data

import ssb64bc.datasets.datasets as datasets
import ssb64bc.datasets.utils as dataset_utils
import ssb64bc.formatting.action_formatters as action_formatters
import ssb64bc.formatting.utils as formatting_utils
import ssb64bc.models.models as models
from ssb64bc.nn.sequence_loss import SequenceLoss

import hparams
import train_utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_filepath',
                        type=str,
                        help="Filepath of train dataset csv file.",
                        required=True)
    parser.add_argument('--val_dataset_filepath',
                        type=str,
                        help="Filepath of val dataset csv file.",
                        required=True)
    parser.add_argument(
        '--exp_directory',
        type=str,
        help="Experiment directory in which to save network weights and other logging information.",
        required=True)
    parser.add_argument('--img_directory',
                        type=str,
                        help="The directory relative to which the image paths are defined.",
                        default="../data/matches")
    parser.add_argument('--load_filepath',
                        type=str,
                        help="If set, load the network state dict from this filepath.",
                        default=None)
    parser.add_argument('--hparams',
                        type=str,
                        help="Comma-separated values to pass to hparams (key=value,...)",
                        default=None)
    return parser


def get_model_and_loss(args, device, n_frames=4):
    """Creates the model and associated loss.
    
    The model and loss are coupled. For example, the multiclass case uses
    cross entropy loss, the multidiscrete requires a custom loss, and the
    recurrent case also requires a sequential loss in addition to the
    underlying loss, which might also need to be customized.
    """
    num_channels = formatting_utils.get_num_channels(args.image_type)
    if args.model_type == "multiclass":
        action_dim = action_formatters.SSB64MulticlassActionFormatter().n_classes
        model = models.MultiframeMulticlassActionModel(num_classes=action_dim,
                                                       n_frames=n_frames,
                                                       n_channels=num_channels)
        criterion = train_utils.get_multiclass_loss(device, args.train_dataset_filepath)
    elif args.model_type == "multidiscrete":
        n_classes_per_action = action_formatters.SSB64MultiDiscreteActionFormatter.N_CLASSES
        model = models.MultiframeMultidiscreteActionModel(n_classes_per_action=n_classes_per_action,
                                                          n_frames=n_frames,
                                                          n_channels=num_channels)
        criterion = train_utils.get_multidiscrete_loss(n_classes_per_action, device,
                                                       args.train_dataset_filepath)
    elif args.model_type == "recurrent_multiclass":
        action_dim = action_formatters.SSB64MulticlassActionFormatter().n_classes
        model = models.RecurrentMulticlassActionModel(output_dim=action_dim,
                                                      dropout_prob=args.recurrent_dropout_prob)
        criterion = train_utils.get_multiclass_loss(device, args.train_dataset_filepath)
        criterion = SequenceLoss(criterion)
    else:
        raise ValueError("Invalid model type: {}".format(args.model_type))

    # Load network weights if provided.
    if args.load_filepath:
        assert os.path.exists(args.load_filepath)
        model.load_state_dict(torch.load(args.load_filepath))

    model = model.to(device)
    return model, criterion


def _get_dataset(dataset_filepath, img_directory, image_type, obs_transform, debug_size, dataset_type,
                 model_type):
    assert os.path.exists(dataset_filepath)

    if model_type == "multiclass":
        action_transform = lambda x: np.argmax(np.array(x))
    else:
        action_transform = lambda x: x

    if dataset_type == "preprocessed":
        # Action transform required because actions stored as one-hot in dataset.
        dataset = datasets.PreprocessedMultiframeDataset(dataset_filepath,
                                                         img_directory,
                                                         transform=None,
                                                         action_transform=action_transform,
                                                         debug_size=debug_size)
    elif dataset_type == "images":
        assert os.path.exists(img_directory)
        dataset = datasets.MultiframeDataset(dataset_filepath,
                                             img_directory,
                                             image_type=image_type,
                                             transform=obs_transform,
                                             action_transform=action_transform,
                                             debug_size=debug_size)
    elif dataset_type == "hdf5":
        dataset = datasets.HDF5Dataset(dataset_filepath, debug_size=debug_size)
    else:
        raise ValueError("Invalid dataset type: {}".format(dataset_type))
    return dataset


def get_data_loaders(args):
    mean, std = dataset_utils.get_image_mean_std(args.image_type)
    obs_transform = dataset_utils.get_image_transforms(mean, std)
    num_workers = train_utils.get_num_workers()

    train_dataset = _get_dataset(args.train_dataset_filepath, args.img_directory, args.image_type,
                                 obs_transform, args.debug_size, args.dataset_type, args.model_type)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               num_workers=num_workers)
    val_dataset = _get_dataset(args.val_dataset_filepath, args.img_directory, args.image_type, obs_transform,
                               args.val_debug_size, args.dataset_type, args.model_type)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    return train_loader, val_loader


def get_optimizer(args, model):
    if args.opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.opt_lr, weight_decay=args.opt_weight_decay)
    elif args.opt_type == "sgd":
        return torch.optim.SGD(model.parameters(),
                               lr=args.opt_lr,
                               momentum=args.opt_momentum,
                               weight_decay=args.opt_weight_decay)
    else:
        raise ValueError("Invalid optimizer: {}".format(args.opt_type))


def main():
    args = get_parser().parse_args()
    args = hparams.merge_args_hparams(args)
    train_utils.save_json(os.path.join(args.exp_directory, "config.json"), args.__dict__)
    train_utils.log_training_information(args)

    # Build data loaders, model, and optimizer.
    device = train_utils.get_device()
    train_loader, val_loader = get_data_loaders(args)
    model, criterion = get_model_and_loss(args, device)
    optimizer = get_optimizer(args, model)

    # Build trainer and validator.
    trainer = train_utils.create_supervised_trainer(model,
                                                    optimizer,
                                                    criterion,
                                                    metrics=train_utils.get_metrics(args, criterion),
                                                    device=device,
                                                    grad_clip=args.grad_clip,
                                                    grad_norm=args.grad_norm)
    validator = train_utils.create_supervised_evaluator(model,
                                                        metrics=train_utils.get_metrics(args, criterion),
                                                        device=device)

    # Add handlers.
    action_names = train_utils.load_action_names(args)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_utils.val_epoch_completed_logger, validator,
                              val_loader, action_names)
    trainer.add_event_handler(Events.EPOCH_STARTED, train_utils.train_epoch_started_logger, args.max_epochs)
    timer = train_utils.attach_timer(trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_utils.train_epoch_completed_logger, timer)
    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              train_utils.batch_logger,
                              n_batches=len(train_loader),
                              split="Train")
    train_utils.maybe_attach_lr_scheduler_handler(trainer, args, optimizer, len(train_loader))
    logger = train_utils.attach_tensorboard_handler(args, trainer, validator, optimizer, model, train_loader)
    validator.add_event_handler(Events.EPOCH_COMPLETED, train_utils.get_model_checkpoint_handler(args),
                                {"model": model})
    validator.add_event_handler(Events.EPOCH_COMPLETED,
                                train_utils.get_interval_model_checkpoint_handler(args), {"model": model})
    validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        train_utils.get_early_stopping_handler(args, trainer, args.early_stopping_patience))
    validator.add_event_handler(Events.ITERATION_COMPLETED,
                                train_utils.batch_logger,
                                n_batches=len(val_loader),
                                split="Val")

    # Run training.
    trainer.run(train_loader, max_epochs=args.max_epochs)
    logger.close()


if __name__ == "__main__":
    main()
