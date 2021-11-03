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
import dwi_ml.experiment.parameter_description as params_d_dwiml
from dwi_ml.experiment.timer import Timer
from dwi_ml.training.training_utils import parse_args_train_model, \
    prepare_data, prepare_batch_sampler_1i_pv, check_unused_args_for_checkpoint
from dwi_ml.utils import format_dict_to_str
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from Learn2Track.checks_for_experiment_parameters import (
    check_all_experiment_parameters)
from Learn2Track.models.learn2track_model import Learn2TrackModel
from Learn2Track.training.trainers import Learn2TrackTrainer
import Learn2Track.training.parameter_description as params_d


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


def prepare_model(input_size, model_params):
    """
    Instantiate model
    """
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        logging.info("Input size inferred from the data: {}"
                     .format(input_size))
        model = Learn2TrackModel(input_size=input_size, **model_params)
        logging.info("Learn2track model instantiated with attributes: \n" +
                     format_dict_to_str(model.attributes))

    return model


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

    # Instantiate everything from checkpoint_state
    # toDo : Check that training set and validation set had the same params
    dataset = prepare_data(checkpoint_state['dataset_params'])
    (training_batch_sampler,
     validation_batch_sampler) = prepare_batch_sampler_1i_pv(
        dataset,
        checkpoint_state['train_sampler_params'],
        checkpoint_state['valid_sampler_params'])
    input_size = checkpoint_state['model_params']['input_size']
    model = prepare_model(input_size, checkpoint_state['model_params'])

    # Instantiate trainer
    with Timer("\n\nPreparing trainer",
               newline=True, color='red'):
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

    # Prepare the trainer from params
    dataset = prepare_data(dataset_params)
    (training_batch_sampler,
     validation_batch_sampler) = prepare_batch_sampler_1i_pv(
        dataset, sampler_params, sampler_params)
    input_group_idx = dataset.volume_groups.index(args.input_group)
    input_size = dataset.nb_features[input_group_idx]
    model = prepare_model(input_size, model_params)

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

    # Initialize logger
    logging.basicConfig(level=args.logging.upper())

    # Verify if a checkpoint has been saved. Else create an experiment.
    if path.exists(os.path.join(args.experiment_path, args.experiment_name,
                                "checkpoint")):
        trainer = init_from_checkpoint(args)
    else:
        trainer = init_from_args(p, args)

    # Run (or continue) the experiment
    try:
        with Timer("\n\n****** Running model!!! ********",
                   newline=True, color='magenta'):
            trainer.run_model()
    except EarlyStoppingError as e:
        print(e)

    trainer.save_model()

    print("Script terminated successfully. Saved experiment in folder : ")
    print(trainer.experiment_path)


if __name__ == '__main__':
    main()
