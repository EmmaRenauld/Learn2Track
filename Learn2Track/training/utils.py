# -*- coding: utf-8 -*-
from dwi_ml.training.utils.trainer import \
    add_training_args as add_training_args_super


def add_training_args(p):
    training_group = add_training_args_super(p)
    training_group.add_argument(
        '--clip_grad', type=float, default=0,
        help="Value to which the gradient norms to avoid exploding gradients."
             "Default = 0 (not clipping).")
