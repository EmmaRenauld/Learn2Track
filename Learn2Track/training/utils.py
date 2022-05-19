# -*- coding: utf-8 -*-
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.utils.trainer import \
    add_training_args as add_training_args_super

from Learn2Track.training.trainers import Learn2TrackTrainer


def add_training_args(p):
    training_group = add_training_args_super(p)
    training_group.add_argument(
        '--clip_grad', type=float, default=0,
        help="Value to which the gradient norms to avoid exploding gradients."
             "Default = 0 (not clipping).")
