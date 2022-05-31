#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model for Learn2Track
"""
import argparse
import logging
import os

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.data.dataset.utils import (
    add_dataset_args, prepare_multisubjectdataset)
from dwi_ml.experiment_utils.prints import add_logging_arg
from dwi_ml.models.utils.direction_getters import (
    add_direction_getter_args, check_args_direction_getter)
from dwi_ml.training.utils.batch_samplers import (
    add_args_batch_sampler, prepare_batchsamplers_train_valid)
from dwi_ml.training.utils.batch_loaders import (
    add_args_batch_loader, prepare_batchloadersoneinput_train_valid)
from dwi_ml.training.utils.experiment import (
    add_mandatory_args_training_experiment,
    add_memory_args_training_experiment)
from dwi_ml.training.utils.trainer import run_experiment

from Learn2Track.models.utils import add_model_args, prepare_model
from Learn2Track.training.trainers import Learn2TrackTrainer
from Learn2Track.training.utils import add_training_args


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_mandatory_args_training_experiment(p)
    add_memory_args_training_experiment(p)
    add_dataset_args(p)
    add_args_batch_sampler(p)
    add_args_batch_loader(p)
    add_training_args(p)

    # Specific to Learn2track:
    add_model_args(p)
    add_direction_getter_args(p)

    add_logging_arg(p)

    return p


def init_from_args(args):
    # Prepare the dataset
    dataset = prepare_multisubjectdataset(args, load_testing=False)

    # Preparing the model

    # (Direction getter)
    if not args.dg_dropout and args.dropout:
        args.dg_dropout = args.dropout
    # (Neighborhood)
    dg_args = check_args_direction_getter(args)
    if args.grid_radius:
        args.neighborhood_radius = args.grid_radius
        args.neighborhood_type = 'grid'
    elif args.sphere_radius:
        args.neighborhood_radius = args.sphere_radius
        args.neighborhood_type = 'axes'
    else:
        args.neighborhood_radius = None
        args.neighborhood_type = None
    # (Nb features)
    input_group_idx = dataset.volume_groups.index(args.input_group_name)
    args.nb_features = dataset.nb_features[input_group_idx]
    # Final model
    model = prepare_model(args, dg_args)

    # Setting log level to INFO maximum for sub-loggers, else it become ugly
    sub_loggers_level = args.logging
    if args.logging == 'DEBUG':
        sub_loggers_level = 'INFO'

    # Preparing the batch samplers
    args.wait_for_gpu = args.use_gpu
    training_batch_sampler, validation_batch_sampler = \
        prepare_batchsamplers_train_valid(dataset, args, args,
                                          sub_loggers_level)

    # Preparing the batch loaders
    args.neighborhood_points = model.neighborhood_points
    training_batch_loader, validation_batch_loader = \
        prepare_batchloadersoneinput_train_valid(dataset, args, args,
                                                 sub_loggers_level)

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = Learn2TrackTrainer(
            model, args.experiments_path, args.experiment_name,
            training_batch_sampler, training_batch_loader,
            validation_batch_sampler, validation_batch_loader,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rate=args.learning_rate, max_epochs=args.max_epochs,
            max_batches_per_epoch=args.max_batches_per_epoch,
            patience=args.patience, from_checkpoint=False,
            weight_decay=args.weight_decay, clip_grad=args.clip_grad,
            # MEMORY
            nb_cpu_processes=args.processes, use_gpu=args.use_gpu,
            log_level=args.logging)
        logging.info("Trainer params : " + format_dict_to_str(trainer.params))

    return trainer


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Setting root logger with high level but we will set trainer to
    # user-defined level.
    logging.basicConfig(level=logging.WARNING)

    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file])
    assert_outputs_exist(p, args, args.experiments_path)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if os.path.exists(os.path.join(args.experiments_path, args.experiment_name,
                                   "checkpoint")):
        raise FileExistsError("This experiment already exists. Delete or use "
                              "script l2t_resume_training_from_checkpoint.py.")

    trainer = init_from_args(args)

    run_experiment(trainer, args.logging)


if __name__ == '__main__':
    main()
