# -*- coding: utf-8 -*-
"""
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca),
        refactored by Emmanuelle Renauld
"""
import logging
from typing import Union

import torch

from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput

from Learn2Track.models.learn2track_model import Learn2TrackModel

logger = logging.getLogger('trainer_logger')


class Learn2TrackTrainer(DWIMLTrainerOneInput):
    """
    Trainer for Learn2Track. Nearly the same as in dwi_ml, but we add the
    clip_grad parameter to avoid exploding gradients, typical in RNN.
    """
    def __init__(self,
                 model: Learn2TrackModel, experiments_path: str,
                 experiment_name: str,
                 batch_sampler: DWIMLBatchIDSampler,
                 batch_loader: DWIMLBatchLoaderOneInput,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01, max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: Union[int, None] = 1000,
                 patience: int = None,
                 nb_cpu_processes: int = 0, use_gpu: bool = False,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False, clip_grad: float = 0,
                 log_level=logging.WARNING):
        """
        Init trainer.

        Additionnal values compared to super:
        clip_grad : float
            The value to which to clip gradients after the backward pass.
            There is no good value here. Default: 1000.
        """
        model_uses_streamlines = True
        self.clip_grad = clip_grad

        super().__init__(model, experiments_path, experiment_name,
                         batch_sampler, batch_loader, model_uses_streamlines,
                         learning_rate, weight_decay, max_epochs,
                         max_batches_per_epoch_training,
                         max_batches_per_epoch_validation,
                         patience,
                         nb_cpu_processes, use_gpu, comet_workspace,
                         comet_project, from_checkpoint, log_level)

    @property
    def params_for_checkpoint(self):
        # We do not need the model_uses_streamlines params, we know it is true
        params = super().params_for_checkpoint
        del params['model_uses_streamlines']
        params.update({
            'clip_grad': self.clip_grad
        })
        return params

    @classmethod
    def init_from_checkpoint(
            cls, model: Learn2TrackModel, experiments_path, experiment_name,
            batch_sampler: DWIMLBatchIDSampler,
            batch_loader: DWIMLBatchLoaderOneInput,
            checkpoint_state: dict, new_patience,
            new_max_epochs, log_level):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).
        """

        # Use super's method but return this learn2track trainer as 'cls'.
        experiment = super(cls, cls).init_from_checkpoint(
            model, experiments_path, experiment_name,
            batch_sampler, batch_loader,
            checkpoint_state, new_patience, new_max_epochs, log_level)

        return experiment

    def _prepare_checkpoint_info(self) -> dict:
        checkpoint_state = super()._prepare_checkpoint_info()
        checkpoint_state['params_for_init'].update({
            'clip_grad': self.clip_grad
        })
        return checkpoint_state

    def fix_parameters(self):
        """
        In our case, clipping gradients to avoid exploding gradients in RNN
        """
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.clip_grad)
