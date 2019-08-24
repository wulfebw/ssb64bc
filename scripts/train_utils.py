import datetime
import json
import multiprocessing as mp
import os
import sys
import textwrap

import h5py
import ignite
import ignite.contrib.handlers.tensorboard_logger as tb_logger
from ignite.engine import Events
import numpy as np
import pandas as pd
import torch
import torchvision

from ssb64bc.nn.multi_discrete_cross_entropy_loss import MultiDiscreteCrossEntropyLoss
from ssb64bc.formatting.utils import get_act_cols, get_act_counts


def get_device():
    # Uses gpu is available.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_num_workers():
    # Returns number of cpus, not the number we can use, but better than nothing.
    return mp.cpu_count() - 1


"""Logging utils."""


def get_log_prefix():
    return "[{}] ".format(str(datetime.datetime.now()))


def log_dataset_information(filepath, split, debug_size):
    df = pd.read_csv(filepath)
    if debug_size is not None:
        df = df.iloc[:debug_size]
    action_counts = get_act_counts(df)
    print(split)
    print(action_counts)
    print()


def log_training_information(args):
    if args.train_dataset_filepath.endswith(".csv"):
        log_dataset_information(args.train_dataset_filepath, "Train", debug_size=args.debug_size)
        log_dataset_information(args.val_dataset_filepath, "Validation", debug_size=args.debug_size)


def load_action_names(args):
    if args.train_dataset_filepath.endswith("csv"):
        df = pd.read_csv(args.train_dataset_filepath)
        return list(get_act_cols(df))
    else:
        return []


def save_json(filepath, d):
    with open(filepath, "w") as outfile:
        json.dump(d, outfile, indent=2)


"""Dataset and weights utils."""


def get_class_weights_df(dataset_filepath, max_weight=10):
    """Get weights for different actions to correct for imbalance.

    Actions that do not occur have a weight of 1.

    Weights are clipped to be smaller than max_weight.
    """
    df = pd.read_csv(dataset_filepath)
    action_keys = get_act_cols(df)
    df = df[action_keys]
    class_sums = df.sum(axis=0)
    max_sum = max(class_sums)
    class_sums[class_sums == 0] = -1
    weights = max_sum / class_sums
    weights = np.clip(weights, 0, max_weight)
    return torch.tensor(weights, dtype=torch.float)


def get_class_weights_hdf5(dataset_filepath, max_weight=10):
    h5file = h5py.File(dataset_filepath)
    actions = np.array(h5file["actions"])
    axes = tuple(range(len(actions.shape) - 1))
    class_sums = actions.sum(axis=axes)
    max_sum = max(class_sums)
    # Set classes that do not occur to have zero weight by making their weight negative then clipping it.
    class_sums[class_sums == 0] = -1
    weights = max_sum / class_sums
    weights = np.clip(weights, 0, max_weight)
    return torch.tensor(weights, dtype=torch.float)


def get_class_weights(dataset_filepath, max_weight=10):
    if dataset_filepath.endswith(".csv"):
        return get_class_weights_df(dataset_filepath, max_weight)
    elif dataset_filepath.endswith(".hdf5"):
        return get_class_weights_hdf5(dataset_filepath, max_weight)
    else:
        raise ValueError("Invalid dataset filepath: {}".format(dataset_filepath))


def get_multiclass_loss(device, dataset_filepath=None):
    """Get the loss used in the multiclass case.

    Args:
        device: The device to place weights if computed.
        dataset_filepath: Filepath to the dataset used for training.
            The dataset will be used to create class weights if provided.

    Returns:
        The multiclass loss criterion.
    """
    weights = None
    if dataset_filepath is not None:
        assert os.path.exists(dataset_filepath)
        weights = get_class_weights(dataset_filepath)
        print("\nclass weights:\n{}".format(weights))
        weights = weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    return criterion


def get_multidiscrete_loss(n_classes_per_action, device, dataset_filepath):
    # TODO: Weight the loss.
    edges = [0] + list(np.cumsum(n_classes_per_action).astype(int))
    return MultiDiscreteCrossEntropyLoss(edges)


"""Ignite utils."""


def get_metrics(args, criterion):
    if args.model_type == "multiclass":
        precision = ignite.metrics.Precision(average=False)
        recall = ignite.metrics.Recall(average=False)
        f1 = ((precision * recall * 2) / (precision + recall + 1e-8)).mean()
        return dict(loss=ignite.metrics.Loss(criterion),
                    accuracy=ignite.metrics.Accuracy(),
                    top_5_accuracy=ignite.metrics.TopKCategoricalAccuracy(k=5),
                    precision=precision,
                    recall=recall,
                    f1=f1)
    elif args.model_type == "multidiscrete":
        return dict(loss=ignite.metrics.Loss(criterion))
    else:
        raise ValueError("invalid model type: {}".format(args.model_type))


def _prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (ignite.utils.convert_tensor(x, device=device, non_blocking=non_blocking),
            ignite.utils.convert_tensor(y, device=device, non_blocking=non_blocking))


def _metrics_transform(output):
    if len(output[1].shape) > 2:
        batch_size, max_seq_len = output[1].shape[:2]
        return output[1].view(batch_size * max_seq_len, -1), output[2].view(batch_size * max_seq_len)
    return output[1], output[2]


def create_supervised_trainer(model,
                              optimizer,
                              loss_fn,
                              metrics={},
                              device=None,
                              grad_clip=1.0,
                              grad_norm=1000.0):
    if device is not None:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        # Don't change the order of these gradient rescaling operations.
        torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), grad_clip)
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        return loss.item(), y_pred, y

    engine = ignite.engine.Engine(_update)

    for name, metric in metrics.items():
        metric._output_transform = _metrics_transform
        metric.attach(engine, name)

    return engine


def create_supervised_evaluator(model, metrics={}, device=None):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=device)
            y_pred = model(x)
            return x, y_pred, y

    engine = ignite.engine.Engine(_inference)

    for name, metric in metrics.items():
        metric._output_transform = _metrics_transform
        metric.attach(engine, name)

    return engine


"""Callbacks for ignite."""


def format_tensor_metric(values):
    return " ".join(["{:.2f}".format(v) for v in values])


def get_metrics_log_str(metrics, split):
    log_str = ""
    for k, v in metrics.items():
        if isinstance(v, float):
            log_str += "\n{} {}: {:.4f}".format(split, k, v)
        elif isinstance(v, torch.Tensor):
            log_str += "\n{} {}: {}".format(split, k, format_tensor_metric(v))
        else:
            log_str += "\n{} {}: {}".format(split, k, v)
    return log_str


def val_epoch_completed_logger(engine, validator, val_loader, action_names):
    print()  # Newline to differentiate train and val iteration loggers.
    validator.run(val_loader)
    log_str = "\n{}".format(action_names)
    log_str += get_metrics_log_str(validator.state.metrics, "Validation")
    log_str = textwrap.indent(log_str, get_log_prefix())
    print(log_str)


def train_epoch_started_logger(engine, max_epochs):
    log_str = "\nEpoch {} / {}".format(engine.state.epoch, max_epochs)
    log_str = textwrap.indent(log_str, get_log_prefix())
    print(log_str)
    print("#" * 60)


def train_epoch_completed_logger(engine, timer=None):
    log_str = get_metrics_log_str(engine.state.metrics, "Train")

    # Adding timing data if available.
    if timer is not None:
        log_str += "\nAverage epoch time: {:.4f} sec\n".format(timer.value())

    log_str = textwrap.indent(log_str.strip(), get_log_prefix())
    print(log_str)
    print("#" * 60)


def batch_logger(engine, n_batches, split):
    itr = (engine.state.iteration % n_batches)
    if itr == 0:
        itr = n_batches
    log_str = "\r{} batch: {} / {}".format(split, itr, n_batches)
    sys.stdout.write(log_str)


def score_function(engine):
    # Larger is better for score function so return negative.
    return -engine.state.metrics.get("loss", 1e6)


def get_model_checkpoint_handler(args):
    os.makedirs(args.exp_directory, exist_ok=True)
    return ignite.handlers.ModelCheckpoint(os.path.join(args.exp_directory, "networks"),
                                           filename_prefix="ssb64",
                                           score_function=score_function,
                                           score_name="loss",
                                           n_saved=10,
                                           require_empty=False)


def get_interval_model_checkpoint_handler(args):
    os.makedirs(args.exp_directory, exist_ok=True)
    return ignite.handlers.ModelCheckpoint(os.path.join(args.exp_directory, "networks"),
                                           filename_prefix="regular",
                                           save_interval=5,
                                           n_saved=100,
                                           require_empty=False)


def get_early_stopping_handler(args, trainer, patience=20):
    return ignite.handlers.EarlyStopping(patience, score_function=score_function, trainer=trainer)


def maybe_attach_lr_scheduler_handler(engine, args, optimizer, n_batches):
    if n_batches <= 1:
        print("Skipping learning rate scheduler due to batch size")
        return
    start_value = args.opt_lr
    end_value = args.opt_lr * args.lr_cycle_ratio
    handler = ignite.contrib.handlers.param_scheduler.CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value,
        end_value,
        cycle_size=n_batches,
        start_value_mult=args.lr_cycle_decay,
        end_value_mult=args.lr_cycle_decay)
    engine.add_event_handler(Events.ITERATION_COMPLETED, handler)


def attach_timer(engine):
    timer = ignite.handlers.Timer(average=True)
    timer.attach(engine, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)
    return timer


def log_random_images(writer, images, num_samples_to_log=5):
    n_samples = len(images)
    idxs = np.random.randint(n_samples, size=num_samples_to_log)
    for i, idx in enumerate(idxs):
        sample_images = images[idx]
        if len(sample_images.shape) < 4:
            sample_images = sample_images.unsqueeze(dim=1)
        grid = torchvision.utils.make_grid(sample_images, nrow=2, normalize=True)
        writer.add_image("sample_{}".format(i), grid)


class GradsScalarHandler(ignite.contrib.handlers.base_logger.BaseWeightsScalarHandler):
    """Copied from ignite with a small change to prevent an exception when used with weight drop."""

    def __init__(self, model, reduction=torch.norm):
        super(GradsScalarHandler, self).__init__(model, reduction)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, tb_logger.TensorboardLogger):
            raise RuntimeError("Handler 'GradsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            name = name.replace('.', '/')
            logger.writer.add_scalar("grads_{}/{}".format(self.reduction.__name__, name),
                                     self.reduction(p.grad), global_step)


def attach_tensorboard_handler(args, trainer, validator, optimizer, model, loader):
    os.makedirs(args.exp_directory, exist_ok=True)
    log_dir = os.path.join(args.exp_directory, "tb")

    metric_names_to_log = ["loss", "accuracy", "top_5_accuracy", "f1"]

    logger = tb_logger.TensorboardLogger(log_dir=log_dir)
    logger.attach(trainer,
                  log_handler=tb_logger.OutputHandler(tag="train", metric_names=metric_names_to_log),
                  event_name=Events.ITERATION_COMPLETED)
    logger.attach(trainer,
                  log_handler=tb_logger.OptimizerParamsHandler(optimizer),
                  event_name=Events.ITERATION_COMPLETED)
    logger.attach(trainer,
                  log_handler=GradsScalarHandler(model, reduction=torch.norm),
                  event_name=Events.ITERATION_COMPLETED)
    logger.attach(validator,
                  log_handler=tb_logger.OutputHandler(tag="validation",
                                                      metric_names=metric_names_to_log,
                                                      another_engine=trainer),
                  event_name=Events.EPOCH_COMPLETED)
    log_random_images(logger.writer, next(iter(loader))[0])

    return logger
