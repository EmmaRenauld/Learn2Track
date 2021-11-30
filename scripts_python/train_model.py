#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model for Learn2Track

See an example of the yaml file in the parameters folder.
"""
import logging
import os
from os import path

import yaml

from dwi_ml.experiment.monitoring import EarlyStoppingError
from dwi_ml.experiment.timer import Timer
from dwi_ml.training.batch_samplers import \
    BatchStreamlinesSamplerOneInputAndPD
from dwi_ml.training.training_utils import parse_args_train_model, \
    prepare_data, check_unused_args_for_checkpoint
from dwi_ml.utils import format_dict_to_str
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from Learn2Track.training.checks_for_tracking_parameters import (
    check_all_experiment_parameters)
from Learn2Track.models.learn2track_model import Learn2TrackModel
from Learn2Track.training.trainers import Learn2TrackTrainer
import Learn2Track.training.parameter_description as params_d
import Learn2Track.training.parameter_description_dwiml as params_d_dwiml


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


def prepare_batchsamplers(dataset, train_sampler_params,
                          valid_sampler_params, model):
    """
    Instantiate a batch sampler (one for training + one for validation).

    There could be a discussion about merging the training and validation
    samplers, but user could potentially set params differently between
    training and validation (ex, more or less noise), so keeping two instances.

    Returns None if the dataset has no training subjects or no validation
    subjects, respectively.
    """
    with Timer("\nPreparing batch samplers...", newline=True, color='green'):
        if dataset.training_set.nb_subjects > 0:
            train_batch_sampler = BatchStreamlinesSamplerOneInputAndPD(
                dataset.training_set, model=model, **train_sampler_params)
            logging.info(
                "\nTraining batch sampler user-defined parameters: \n" +
                format_dict_to_str(train_batch_sampler.params))
        else:
            train_batch_sampler = None

        if dataset.validation_set.nb_subjects > 0:
            valid_batch_sampler = BatchStreamlinesSamplerOneInputAndPD(
                dataset.validation_set, model=model, **valid_sampler_params)
            logging.info(
                "\nValidation batch sampler user-defined parameters: \n" +
                format_dict_to_str(valid_batch_sampler.params))
        else:
            valid_batch_sampler = None

    return train_batch_sampler, valid_batch_sampler


def init_from_checkpoint(args):
    check_unused_args_for_checkpoint(args, ['input_group', 'target_group'])

    # Loading checkpoint
    checkpoint_state = Learn2TrackTrainer.load_params_from_checkpoint(
        args.experiment_path,
        args.experiment_name)

    # Stop now if early stopping was triggered.
    Learn2TrackTrainer.check_stopping_cause(
        checkpoint_state,
        args.override_checkpoint_patience,
        args.override_checkpoint_max_epochs)

    # Prepare data
    dataset = prepare_data(checkpoint_state['train_data_params'])
    # toDo Verify that valid dataset is the same.
    #  checkpoint_state['valid_data_params']

    # Prepare model
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = Learn2TrackModel.init_from_checkpoint(
            **checkpoint_state['model_params'])
        logging.info("Learn2track model user-defined parameters: \n" +
                     format_dict_to_str(model.params))
        logging.info("Learn2track model final parameters:" +
                     format_dict_to_str(model.params_per_layer))

    # Prepare batch samplers
    (training_batch_sampler,
     validation_batch_sampler) = prepare_batchsamplers(
        dataset,
        checkpoint_state['train_sampler_params'],
        checkpoint_state['valid_sampler_params'], model)

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = Learn2TrackTrainer.init_from_checkpoint(
            training_batch_sampler,
            validation_batch_sampler,
            model,
            checkpoint_state,
            args.override_checkpoint_patience,
            args.override_checkpoint_max_epochs)
    return trainer


def init_from_args(p, args):
    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file, args.yaml_parameters])
    assert_outputs_exist(p, args, args.experiment_path)

    # Load parameters
    with open(args.yaml_parameters) as f:
        yaml_parameters = yaml.safe_load(f.read())

    # Perform checks
    (sampling_params, training_params, model_params, memory_params,
     randomization) = check_all_experiment_parameters(yaml_parameters)

    # Modifying params to copy the checkpoint_state params.
    # Params coming from the yaml file must have the same keys as when
    # using a checkpoint.
    experiment_params = {
        'hdf5_file': args.hdf5_file,
        'experiment_path': args.experiment_path,
        'experiment_name': args.experiment_name,
        'comet_workspace': args.comet_workspace,
        'comet_project': args.comet_project}

    # MultiSubjectDataset parameters
    dataset_params = {**memory_params,
                      **experiment_params}

    # If you wish to have different parameters for the batch sampler during
    # training and validation, change values below.
    sampler_params = {**sampling_params,
                      **model_params,
                      **randomization,
                      **memory_params,
                      'input_group_name': args.input_group,
                      'streamline_group_name': args.target_group,
                      'wait_for_gpu': memory_params['use_gpu']}

    model_params.update(memory_params)

    # Prepare the dataset
    dataset = prepare_data(dataset_params)

    # Preparing the model
    input_group_idx = dataset.volume_groups.index(args.input_group)
    nb_features = dataset.nb_features[input_group_idx]
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        logging.debug("Nb of features in the data: {}".format(nb_features))
        model = Learn2TrackModel(experiment_name=args.experiment_name,
                                 input_group_name='input',
                                 nb_features=nb_features, **model_params)
        logging.info("Learn2track model user-defined parameters: " +
                     format_dict_to_str(model.params) + '\n')
        logging.info("Learn2track model final parameters:" +
                     format_dict_to_str(model.params_per_layer))

    # Preparing the batch sampler
    (training_batch_sampler,
     validation_batch_sampler) = prepare_batchsamplers(
        dataset, sampler_params, sampler_params, model)

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = Learn2TrackTrainer(
            training_batch_sampler, validation_batch_sampler, model,
            **training_params, **experiment_params, **memory_params,
            from_checkpoint=False)

    return trainer


def main():
    p = parse_args_train_model()
    add_project_specific_args(p)
    args = p.parse_args()

    if args.print_description:
        print(params_d_dwiml.__doc__ + params_d.__doc__)
        exit(0)

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
        if args.resume:
            trainer = init_from_checkpoint(args)
        else:
            raise ValueError("This experiment already exists. Use --resume to "
                             "load previous state from checkpoint and resume "
                             "experiment.")
    else:
        trainer = init_from_args(p, args)
    logging.info("Trainer params : " + format_dict_to_str(trainer.params))

    # Run (or continue) the experiment
    try:
        with Timer("\n\n****** Training and validating model!!! ********",
                   newline=True, color='magenta'):
            trainer.train_and_validate(args.logging_choice)
    except EarlyStoppingError as e:
        print(e)

    trainer.save_model()

    logging.info("Script terminated successfully. \n"
                 "Saved experiment in folder : {}"
                 .format(trainer.experiment_path))
    print("Summary: ran {} epochs. Best loss was {} at epoch #{}"
          .format(trainer.current_epoch + 1,
                  trainer.best_epoch_monitoring.best_value,
                  trainer.best_epoch_monitoring.best_epoch + 1))


if __name__ == '__main__':
    main()
