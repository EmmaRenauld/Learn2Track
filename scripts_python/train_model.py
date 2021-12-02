#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model for Learn2Track

See an example of the yaml file in the parameters folder.
"""
import argparse
import logging
import os
from os import path

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.batch_samplers.utils import (add_args_batch_sampler,
                                         prepare_batchsampleroneinput)
from dwi_ml.data.dataset.utils import (add_args_dataset,
                                       prepare_multisubjectdataset)
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.utils import add_training_args, run_experiment

from Learn2Track.models.learn2track_model import prepare_model


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'experiment_path', default='./', metavar='p',
        help='Path where to save your experiment. \nComplete path will be '
             'experiment_path/experiment_name. Default: ./')
    p.add_argument(
        'experiment_name', metavar='n',
        help='If given, name for the experiment. Else, model will decide the '
             'name to \ngive based on time of day.')

    p.add_argument(
        '--logging', dest='logging_choice',
        choices=['error', 'warning', 'info', 'as_much_as_possible', 'debug'],
        help="Logging level. Error, warning, info are as usual.\n The other "
             "options are two equivalents of 'debug' level. \nWith "
             "'as_much_as_possible', we print the debug level only when the "
             "final \nresult is still readable (even during parallel training "
             "and during tqdm loop). \n'debug' prints everything always, even "
             "if ugly.")

    add_args_dataset(p)
    add_args_batch_sampler(p)
    add_training_args(p)

    return p


def add_project_specific_args(p):
    p.add_argument('--input_group', metavar='i',
                   help='Name of the input group. \n'
                        '**If a checkpoint exists, this information is '
                        'already contained in the \ncheckpoint and is not '
                        'necessary. Else, mandatory.')
    p.add_argument('--target_group', metavar='t',
                   help='Name of the target streamline group. \n'
                        '**If a checkpoint exists, this information is '
                        'already contained in the \ncheckpoint and is not '
                        'necessary. Else, mandatory.')


def init_from_args(p, args):
    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file, args.yaml_parameters])
    assert_outputs_exist(p, args, args.experiment_path)

    # Prepare the dataset
    dataset = prepare_multisubjectdataset(args)

    # Preparing the model
    input_group_idx = dataset.volume_groups.index(args.input_group)
    nb_features = dataset.nb_features[input_group_idx]
    model_params['input_group_idx'] = input_group_idx
    model_params['nb_features'] = nb_features
    model_params['experiment_name'] = args.experiment_name
    model = prepare_model(model_params)

    # Preparing the batch samplers
    with Timer("\nPreparing batch samplers...", newline=True, color='green'):
        logging.info("Training batch sampler...")
        training_batch_sampler = prepare_batchsampleroneinput(
            dataset.training_set, args)

        if dataset.validation_set.nb_subjects > 0:
            logging.info("Validation batch sampler...")
            validation_batch_sampler = prepare_batchsampleroneinput(
                dataset.training_set, args)

        else:
            validation_batch_sampler = None



    return trainer


def main():
    p = prepare_arg_parser()
    add_project_specific_args(p)
    args = p.parse_args()

    # Initialize logger for preparation (loading data, model, experiment)
    # If 'as_much_as_possible', we will modify the logging level when starting
    # the training, else very ugly
    logging_level = args.logging_choice.upper()
    if args.logging_choice == 'as_much_as_possible':
        logging_level = 'DEBUG'
    logging.basicConfig(level=logging_level)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if path.exists(os.path.join(args.experiment_path, args.experiment_name,
                                "checkpoint")):
        raise FileExistsError("This experiment already exists. Delete or use "
                              "script resume_experiment.py.")

    trainer = init_from_args(p, args)

    run_experiment(trainer, args.logging_choice)


if __name__ == '__main__':
    main()
