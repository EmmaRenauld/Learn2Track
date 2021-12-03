# -*- coding: utf-8 -*-
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.utils import add_training_args as add_training_args_super

from Learn2Track.training.trainers import Learn2TrackTrainer


def add_training_args(p):
    training_group = add_training_args_super(p)
    training_group.add_argument(
        '--clip_grad', action='store_true',
        help="Clip the gradient norms to avoid exploding gradients.")


def prepare_trainer(training_batch_sampler, validation_batch_sampler, model,
                    args):
    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = Learn2TrackTrainer(
            training_batch_sampler, validation_batch_sampler, model,
            args.experiment_path, args.experiment_name,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rate=args.learning_rate, max_epochs=args.max_epochs,
            max_batches_per_epoch=args.max_batches_per_epoch,
            patience=args.patience, from_checkpoint=False,
            weight_decay=args.weight_decay, clip_grad=args.clip_grad,
            # MEMORY
            # toDo
            nb_cpu_processes=args.processes,
            taskman_managed=args.taskman_managed, use_gpu=args.use_gpu)
        logging.info("Trainer params : " + format_dict_to_str(trainer.params))

    return trainer