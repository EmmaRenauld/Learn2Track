# -*- coding: utf-8 -*-
"""
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca),
        refactored by Emmanuelle Renauld
date: 24/08/2021
"""
import logging

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data.dataloader import DataLoader

from dwi_ml.training.batch_samplers import (
    BatchStreamlinesSamplerOneInput as BatchSampler)
from dwi_ml.training.trainers import DWIMLAbstractTrainer

from Learn2Track.models.learn2track_model import Learn2TrackModel


class Learn2TrackTrainer(DWIMLAbstractTrainer):
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
                 batch_sampler_training: BatchSampler,
                 batch_sampler_validation: BatchSampler,
                 model: Learn2TrackModel, experiment_path: str,
                 experiment_name: str, learning_rate: float,
                 weight_decay: float, max_epochs: int,
                 max_batches_per_epoch: int, patience: int,
                 nb_cpu_processes: int, taskman_managed: bool, use_gpu: bool,
                 comet_workspace: str, comet_project: str,
                 from_checkpoint: bool, clip_grad: float):
        """ Init trainer

        Additionnal values compared to super:
        clip_grad : float
            The value to which to clip gradients after the backward pass.
            There is no good value here.
        """
        super().__init__(batch_sampler_training, batch_sampler_validation,
                         model, experiment_path, experiment_name,
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

    def estimate_nb_batches_per_epoch(self):
        logging.info("Learn2track: Estimating training epoch statistics...")
        n_train_batches_capped, _ = self._compute_epoch_stats(
            self.train_batch_sampler)

        n_valid_batches_capped = None
        if self.valid_batch_sampler is not None:
            logging.info("Learn2track: Estimating validation epoch "
                         "statistics...")
            n_valid_batches_capped, _ = self._compute_epoch_stats(
                self.valid_batch_sampler)

        return n_train_batches_capped, n_valid_batches_capped

    # train_validate_and_save_loss  as super
    # train_one_epoch               as super
    # validate_one_epoch            as super

    @classmethod
    def init_from_checkpoint(
            cls, batch_sampler_training: BatchSampler,
            batch_sampler_validation: BatchSampler, model: Learn2TrackModel,
            checkpoint_state: dict, new_patience, new_max_epochs):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).
        """

        # Use super's method but return this learn2track trainer as 'cls'.
        experiment = super(cls, cls).init_from_checkpoint(
            batch_sampler_training, batch_sampler_validation, model,
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

    def _compute_epoch_stats(self, batch_sampler: BatchSampler):
        """
        Compute approximated statistics about epochs.

        Since the exact data weight per batch can vary based on data
        augmentation in the batch sampler, we approximate the epoch stats
        using a sample batch.

        Returns
        -------
        n_batches : int
            Approximate number of updates per epoch
        batch_sequence_size : int
            Approximate number of sequences per batch
        """
        # Use temporary RNG states to preserve random "coherency"
        # e.g. when resuming an experiment
        sampler_rng_state_bk = batch_sampler.np_rng.get_state()

        dataloader = DataLoader(batch_sampler.dataset,
                                batch_sampler=batch_sampler,
                                num_workers=0,
                                collate_fn=batch_sampler.load_batch)

        # Get a sample batch to compute stats
        # Note that using this does not really work the same way as during
        # training. The __iter__ function of the batch sampler is called
        # 5 times, instead of "yielding" 5 times. So the whole memory of
        # streamlines and subjects that have already been used is resettled
        # each time, and there is a possibility that the same streamlines will
        # be sampled more than once. But this is just for stats so ok.
        logging.info("Running the dataloader for 5 iterations, just to "
                     "compute statistics..")
        sample_batches = [next(iter(dataloader))[0] for _ in range(5)]

        # Restore RNG states. toDo OK??? Voir avec Philippe
        batch_sampler.np_rng.set_state(sampler_rng_state_bk)

        if batch_sampler.dataset.is_lazy:
            batch_sampler.dataset.volume_cache_manager = None

        # Compute stats about epoch
        # toDO CHANGE THIS TO COUNT TIMESTEPS INSTEAD OF STREAMLINES
        #  Philippe se créait même un nouveau batch_sampler avec split_ratio=0
        #  mais ici ça fitte pas avec le abstract batch sampler. Donc changer
        #  le compte plutôt.
        logging.warning('(THIS NEEDS DEBUGGING. Check sample_data output and '
                        'see if we can count the number of points correctly)')
        batch_sizes = []
        for sample_data in sample_batches:
            if isinstance(sample_data, PackedSequence):
                batch_sizes.append(sample_data.batch_sizes[0])
            else:
                batch_sizes.append(len(sample_data))
        avg_batch_size = int(np.mean(batch_sizes))
        if avg_batch_size == 0:
            raise ValueError("The allowed batch size ({}) is too small! "
                             "Sampling 0 streamlines per batch."
                             .format(batch_sampler.max_batch_size))
        logging.info("We have computed that in average, each batch has a "
                     "size of ~{} (in number of datapoints)"
                     .format(avg_batch_size))

        # Define the number of batch per epoch
        dataset_size = batch_sampler.dataset.total_nb_streamlines[
            batch_sampler.streamline_group_idx]
        n_batches = int(dataset_size / avg_batch_size)
        n_batches_capped = min(n_batches, self.max_batches_per_epochs)

        logging.info("Dataset had {} streamlines (before data augmentation)\n"
                     "We will be using approximately {} iterations (i.e. "
                     "batches) per epoch (but not more than the allowed {}).\n"
                     .format(dataset_size, n_batches,
                             self.max_batches_per_epochs))

        return n_batches_capped, avg_batch_size

    def run_model(self, batch_inputs, batch_streamlines):
        dirs = self.model.format_directions(batch_streamlines, self.device)

        # Formatting the previous dirs for all points.
        n_prev_dirs = self.model.format_previous_dirs(dirs, self.device)

        # Not keeping the last point: only useful to get the last direction
        # (last target), but won't be used as an input.
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
