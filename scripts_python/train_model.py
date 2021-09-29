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

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment.monitoring import EarlyStoppingError
from dwi_ml.experiment.timer import Timer
from dwi_ml.models.batch_samplers import (
    BatchStreamlinesSampler1IPV as BatchSampler)
from dwi_ml.training.utils import parse_args_train_model
from dwi_ml.utils import format_dict_to_str

from Learn2Track.checks_for_experiment_parameters import (
    check_all_experiment_parameters)
from Learn2Track.models.learn2track_model import Learn2TrackModel
from Learn2Track.training.trainers import Learn2TrackTrainer


def prepare_data_and_model(dataset_params, train_sampler_params,
                           valid_sampler_params, model_params):
    # Instantiate dataset classes
    with Timer("\n\nPreparing testing and validation sets",
               newline=True, color='blue'):
        dataset = MultiSubjectDataset(**dataset_params)
        dataset.load_data()

        logging.info("\n\nDataset attributes: \n" +
                     format_dict_to_str(dataset.attributes))

    # Instantiate batch
    volume_group_name = train_sampler_params['input_group_name']
    streamline_group_name = train_sampler_params['streamline_group_name']
    volume_group_idx = dataset.volume_groups.index(volume_group_name)
    with Timer("\n\nPreparing batch samplers with volume: '{}' and "
               "streamlines '{}'"
               .format(volume_group_name, streamline_group_name),
               newline=True, color='green'):
        # Batch samplers could potentially be set differently between training
        # and validation.
        training_batch_sampler = BatchSampler(dataset.training_set,
                                              **train_sampler_params)
        validation_batch_sampler = BatchSampler(dataset.validation_set,
                                                **valid_sampler_params)
        logging.info("\n\nTraining batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))
        logging.info("\n\nValidation batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))

    # Instantiate model
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        input_size = dataset.nb_features[volume_group_idx]
        logging.info("Input size inferred from the data: {}"
                     .format(input_size))
        model = Learn2TrackModel(input_size=input_size, **model_params)
        logging.info("Learn2track model instantiated with attributes: \n" +
                     format_dict_to_str(model.attributes))

    return training_batch_sampler, validation_batch_sampler, model


def main():
    args = parse_args_train_model()

    # Check that all files exist
    if not path.exists(args.hdf5_file):
        raise FileNotFoundError("The hdf5 file ({}) was not found!"
                                .format(args.hdf5_file))
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
                args.experiment_path, args.experiment_name)
        if args.parameters_filename:
            logging.warning('Resuming experiment from checkpoint. Yaml file '
                            'option was not necessary and will not be used!')
        if args.hdf5_file:
            logging.warning('Resuming experiment from checkpoint. hdf5 file '
                            'option was not necessary and will not be used!')

        # Stop now if early stopping was triggered.
        Learn2TrackTrainer.check_stopping_cause(
            checkpoint_state, args.override_checkpoint_patience,
            args.override_checkpoint_max_epochs)

        # Prepare the trainer and model from checkpoint_state
        # Note that model state will be updated later!
        # toDo : Check that training set and validation set had the same params
        (training_batch_sampler, validation_batch_sampler,
         model) = prepare_data_and_model(
            checkpoint_state['train_data_params'],
            checkpoint_state['train_sampler_params'],
            checkpoint_state['valid_sampler_params'],
            checkpoint_state['model_params'])

        # Instantiate trainer
        with Timer("\n\nPreparing trainer",
                   newline=True, color='red'):
            trainer = Learn2TrackTrainer.init_from_checkpoint(
                training_batch_sampler, validation_batch_sampler, model,
                checkpoint_state, args.override_checkpoint_patience,
                args.override_checkpoint_max_epochs)
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
            'hdf5_file': args.hdf5_file,
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

    model.save(trainer.experiment_path)

    print("Script terminated successfully. Saved experiment in folder : ")
    print(trainer.experiment_path)


if __name__ == '__main__':
    main()
