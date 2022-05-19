# -*- coding: utf-8 -*-
"""
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca),
        refactored by Emmanuelle Renauld
"""
import torch

from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from dwi_ml.training.batch_loaders import BatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput

from Learn2Track.models.learn2track_model import Learn2TrackModel


class Learn2TrackTrainer(DWIMLTrainerOneInput):
    """Trainer for Learn2Track.

    This Trainer class's train() method:
        - Creates DataLoaders from the batch_samplers. Collate_fn will be the
        sampler.load_batch() method, and the dataset will be
        sampler.source_data.
        - Trains each epoch by using compte_batch_loss.

    Comet is used to save training information, but some logs will also be
    saved locally in the experiment_path.
    """
    def __init__(self,
                 model: Learn2TrackModel, experiments_path: str,
                 experiment_name: str,
                 batch_sampler_training: DWIMLBatchSampler,
                 batch_loader_training: BatchLoaderOneInput,
                 batch_sampler_validation: DWIMLBatchSampler = None,
                 batch_loader_validation: BatchLoaderOneInput = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01, max_epochs: int = 10,
                 max_batches_per_epoch: int = 1000, patience: int = None,
                 nb_cpu_processes: int = 0, taskman_managed: bool = False,
                 use_gpu: bool = False, comet_workspace: str = None,
                 comet_project: str = None, from_checkpoint: bool = False,
                 clip_grad: float = 0):
        """ Init trainer

        Additionnal values compared to super:
        clip_grad : float
            The value to which to clip gradients after the backward pass.
            There is no good value here. Default: 1000.
        """
        super().__init__(model, experiments_path, experiment_name,
                         batch_sampler_training, batch_loader_training,
                         batch_sampler_validation, batch_loader_validation,
                         learning_rate, weight_decay, max_epochs,
                         max_batches_per_epoch, patience, nb_cpu_processes,
                         taskman_managed, use_gpu, comet_workspace,
                         comet_project, from_checkpoint)

        self.clip_grad = clip_grad

    @property
    def params(self):
        params = super().params
        params.update({
            'clip_grad': self.clip_grad
        })
        return params

    # init_comet as super
    # estimate_nb_batches_per_epoch
    # train_validate_and_save_loss  as super
    # train_one_epoch               as super
    # validate_one_epoch            as super

    @classmethod
    def init_from_checkpoint(
            cls, model: Learn2TrackModel, experiments_path, experiment_name,
            train_batch_sampler: DWIMLBatchSampler,
            train_batch_loader: BatchLoaderOneInput,
            valid_batch_sampler: DWIMLBatchSampler,
            valid_batch_loader: BatchLoaderOneInput,
            checkpoint_state: dict, new_patience, new_max_epochs):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).
        """

        # Use super's method but return this learn2track trainer as 'cls'.
        experiment = super(cls, cls).init_from_checkpoint(
            model, experiments_path, experiment_name,
            train_batch_sampler, train_batch_loader,
            valid_batch_sampler, valid_batch_loader,
            checkpoint_state, new_patience, new_max_epochs)

        return experiment

    def _prepare_checkpoint_state(self) -> dict:
        checkpoint_state = super()._prepare_checkpoint_state()
        checkpoint_state['params_for_init'].update({
            'clip_grad': self.clip_grad
        })
        return checkpoint_state

    # save_checkpoint_state:       same as super
    # _should quit                 same as user
    # _update_taskman_report       same as user
    # _save_log_from_array         same as user
    # load_params_from_checkpoint  same as user
    # check_early_stopping         same as user

    def run_model(self, batch_inputs, batch_streamlines):
        dirs = self.model.format_directions(batch_streamlines, self.device)

        # Formatting the previous dirs for all points.
        n_prev_dirs = self.model.format_previous_dirs(dirs, self.device)

        # Not keeping the last point: only useful to get the last direction
        # (last target), but won't be used as an input.
        if n_prev_dirs is not None:
            n_prev_dirs = [s[:-1] for s in n_prev_dirs]

        try:
            # Apply model. This calls our model's forward function
            # (the hidden states are not used here, neither as input nor
            # outputs. We need them only during tracking).
            model_outputs, _ = self.model(batch_inputs, n_prev_dirs,
                                          self.device)
        except RuntimeError:
            # Training RNNs with variable-length sequences on the GPU can
            # cause memory fragmentation in the pytorch-managed cache,
            # possibly leading to "random" OOM RuntimeError during
            # training. Emptying the GPU cache seems to fix the problem for
            # now. We don't do it every update because it can be time
            # consuming.
            torch.cuda.empty_cache()
            model_outputs, _ = self.model(batch_inputs, n_prev_dirs,
                                          self.device)

        # Returning the directions too, to be re-used in compute_loss
        # later instead of computing them twice.
        return model_outputs, dirs

    def compute_loss(self, run_model_output_tuple, _):
        # In theory to do like super, 2nd parameters, targets, would contain
        # the batch streamlines and we would do:
        # directions, packed_directions = self.model.format_directions(
        #             batch_streamlines)
        # Choice 2: As this was already computed when running run_model
        # the formatted targets are returned with the model outputs.
        # 2nd params becomes unused.
        model_outputs, targets = run_model_output_tuple

        # Compute loss using model.compute_loss (as in super)
        mean_loss = self.model.compute_loss(model_outputs, targets,
                                            self.device)
        return mean_loss

    def fix_parameters(self):
        """
        In our case, clipping gradients to avoid exploding gradients in RNN
        """
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.clip_grad)
