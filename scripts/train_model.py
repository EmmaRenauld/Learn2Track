#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model for Learn2Track

See and example of the yaml file in the parameters folder.
"""
import argparse
import logging
import os
from os import path
from typing import List

import torch
import yaml

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment.monitoring import EarlyStoppingError
from dwi_ml.experiment.timer import Timer
from dwi_ml.model.batch_samplers import (
    BatchStreamlinesSampler1IPV as BatchSampler)
from dwi_ml.utils import format_dict_to_str

from Learn2Track.checks_for_experiment_parameters import (
    check_all_experiment_parameters)
from Learn2Track.model.learn2track_model import Learn2TrackModel
from Learn2Track.model.stacked_rnn import StackedRNN
from Learn2Track.training.trainers import Learn2TrackTrainer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Path where to save your experiment. Complete path '
                        'will be experiment_path/experiment_name.')
    p.add_argument('--input_group',
                   help='Name of the input group. If a checkpoint exists, '
                        'this information is already contained in the '
                        'checkpoint and is not necessary.')
    p.add_argument('--target_group',
                   help='Name of the target streamline group. If a checkpoint '
                        'exists, this information is already contained in the '
                        'checkpoint and is not necessary.')
    p.add_argument('--hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects. If a '
                        'checkpoint exists, this information is already '
                        'contained in the checkpoint and is not necessary.')
    p.add_argument('--parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for '
                        'an example. If a checkpoint exists, this information '
                        'is already contained in the checkpoint and is not '
                        'necessary.')
    p.add_argument('--experiment_name',
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to give based on time of day.')
    p.add_argument('--override_checkpoint_patience', type=int,
                   help='If a checkpoint exists, patience can be increased '
                        'to allow experiment to continue if the allowed '
                        'number of bad epochs has been previously reached.')
    p.add_argument('--logging', choices=['error', 'warning', 'info', 'debug'],
                   help="Logging level. One of ['error', 'warning', 'info', "
                        "'debug']. Default: Info.")
    p.add_argument('--comet_workspace',
                   help='Your comet workspace. If not set, comet.ml will not '
                        'be used. See our docs/Getting Started for more '
                        'information on comet and its API key.')
    p.add_argument('--comet_project',
                   help='Send your experiment to a specific comet.ml project. '
                        'If not set, it will be sent to Uncategorized '
                        'Experiments.')
    arguments = p.parse_args()

    return arguments


def build_model(input_size: int, nb_previous_dirs: int,
                prev_dirs_embedding_size: int,
                prev_dirs_embedding_cls, input_embedding_cls,
                input_embedding_size_ratio: float, rnn_key: str,
                rnn_layer_sizes: List[int], dropout: float,
                direction_getter_cls, **_):

    # 1. Previous dir embedding
    if nb_previous_dirs > 0:
        prev_dir_embedding_size = prev_dirs_embedding_size * \
                                  nb_previous_dirs
        prev_dir_embedding_model = prev_dirs_embedding_cls(
            input_size=nb_previous_dirs * 3,
            output_size=prev_dir_embedding_size)
    else:
        prev_dir_embedding_model = None
        prev_dir_embedding_size = 0

    # 2. Input embedding
    input_embedding_size = int(input_size * input_embedding_size_ratio)
    input_embedding_model = input_embedding_cls(
        input_size=input_size,
        output_size=input_embedding_size)

    # 3. Stacked RNN
    rnn_input_size = prev_dir_embedding_size + input_embedding_size
    rnn_model = StackedRNN(rnn_key, rnn_input_size, rnn_layer_sizes,
                           use_skip_connections=True,
                           use_layer_normalization=True,
                           dropout=dropout)

    # 4. Direction getter
    direction_getter_model = direction_getter_cls(rnn_model.output_size)

    # 5. Putting all together
    model = Learn2TrackModel(
        previous_dir_embedding_model=prev_dir_embedding_model,
        input_embedding_model=input_embedding_model,
        rnn_model=rnn_model,
        direction_getter_model=direction_getter_model)

    logging.debug("Learn2track model instantiated with attributes: \n" +
                  format_dict_to_str(model.attributes))

    return model


def prepare_data_and_model(dataset_params, train_sampler_params,
                           valid_sampler_params, model_params):
    # Instantiate dataset classes
    with Timer("\n\nPreparing testing and validation sets",
               newline=True, color='blue'):
        logging.debug("Dataset params: " + format_dict_to_str(dataset_params))
        dataset = MultiSubjectDataset(**dataset_params)
        dataset.load_data()

        logging.info("\n\nDataset attributes: \n" +
                     format_dict_to_str(dataset.attributes))

    # Instantiate batch
    volume_group_name = train_sampler_params['input_group_name']
    volume_group_idx = dataset.volume_groups.index(volume_group_name)
    with Timer("\n\nPreparing batch samplers with volume: '{}' and "
               "streamlines '{}'"
               .format(volume_group_name,
                       train_sampler_params['streamline_group_name']),
               newline=True, color='green'):
        logging.debug("Training batch sampler params: " +
                      format_dict_to_str(train_sampler_params))

        training_batch_sampler = BatchSampler(dataset.training_set,
                                              **train_sampler_params)
        validation_batch_sampler = BatchSampler(dataset.validation_set,
                                                **valid_sampler_params)
        logging.info("\n\nTraining batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))
        logging.info("\n\nValidation batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))

    # Instantiate model
    input_size = dataset.nb_features[volume_group_idx]
    with Timer("\n\nPreparing model",
               newline=True, color='yellow'):
        logging.debug("Model params: " + format_dict_to_str(model_params))
        model = build_model(input_size, **model_params)

    return training_batch_sampler, validation_batch_sampler, model


def main():
    args = parse_args()

    # Check that all files exist
    if not path.exists(args.hdf5_filename):
        raise FileNotFoundError("The hdf5 file ({}) was not found!"
                                .format(args.hdf5_filename))
    if not path.exists(args.parameters_filename):
        raise FileNotFoundError("The Yaml parameters file was not found: {}"
                                .format(args.parameters_filename))

    # Initialize logger
    if args.logging:
        level = args.logging.upper()
    else:
        level = 'INFO'
    logging.basicConfig(level=level)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if path.exists(os.path.join(args.experiment_path, args.experiment_name,
                                "checkpoint")):
        # Loading checkpoint
        print("Experiment checkpoint folder exists, resuming experiment!")
        checkpoint_state = \
            Learn2TrackTrainer.load_params_from_checkpoint(
                args.experiment_path)
        if args.parameters_filename:
            logging.warning('Resuming experiment from checkpoint. Yaml file '
                            'option was not necessary and will not be used!')
        if args.hdf5_filename:
            logging.warning('Resuming experiment from checkpoint. hdf5 file '
                            'option was not necessary and will not be used!')

        # Stop now if early stopping was triggered.
        Learn2TrackTrainer.check_early_stopping(
            checkpoint_state, args.override_checkpoint_patience)

        # Prepare the trainer from checkpoint_state
        (training_batch_sampler, validation_batch_sampler,
         model) = prepare_data_and_model(
            checkpoint_state['dataset_params'],
            checkpoint_state['train_sampler_params'],
            checkpoint_state['valid_sampler_params'],
            checkpoint_state['model_params'])

        # Instantiate trainer
        with Timer("\n\nPreparing trainer",
                   newline=True, color='red'):
            trainer = Learn2TrackTrainer.init_from_checkpoint(
                training_batch_sampler, validation_batch_sampler, model,
                checkpoint_state)
    else:
        # Load parameters
        with open(args.parameters_filename) as f:
            yaml_parameters = yaml.safe_load(f.read())

        # Perform checks
        # We have decided to use yaml for a more rigorous way to store
        # parameters, compared, say, to bash. However, no argparser is used so
        # we need to make our own checks.
        (sampling_params, training_params, model_params, memory_params,
         randomization) = check_all_experiment_parameters(yaml_parameters)

        # Modifying params to copy the checkpoint_state params.
        # Params coming from the yaml file must have the same keys as when
        # using a checkpoint.
        experiment_params = {
            'hdf5_filename': args.hdf5_filename,
            'experiment_path': args.experiment_path,
            'experiment_name': args.experiment_name,
            'comet_workspace': args.comet_workspace,
            'comet_project': args.comet_project}

        # MultiSubjectDataset parameters
        dataset_params = {**memory_params,
                          **experiment_params}

        # If you wish to have different parameters for the batch sampler during
        # trainnig and validation, change values below.
        sampler_params = {**sampling_params,
                          **model_params,
                          **randomization,
                          **memory_params,
                          'input_group_name': args.input_group,
                          'streamline_group_name': args.target_group,
                          'wait_for_gpu': memory_params['use_gpu']}

        model_params.update(memory_params)

        # Prepare the trainer from params
        (training_batch_sampler, validation_batch_sampler,
         model) = prepare_data_and_model(dataset_params, sampler_params,
                                         sampler_params, model_params)

        # Instantiate trainer
        with Timer("\n\nPreparing trainer", newline=True, color='red'):
            trainer = Learn2TrackTrainer(
                training_batch_sampler, validation_batch_sampler, model,
                **training_params, **experiment_params, **memory_params,
                from_checkpoint=False)

    #####
    # Run (or continue) the experiment
    #####
    try:
        with Timer("\n\n****** Running model!!! ********",
                   newline=True, color='magenta'):
            trainer.run_model()
    except EarlyStoppingError as e:
        print(e)

    trainer.save_model()

    print("Script terminated successfully. Saved experiment in folder : ")
    print(trainer.experiment_dir)


if __name__ == '__main__':
    main()
