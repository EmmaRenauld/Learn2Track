# -*- coding: utf-8 -*-
"""
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca),
        refactored by Emmanuelle Renauld
"""
import logging

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

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
                 model: Learn2TrackModel, experiment_path: str,
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
        super().__init__(model, experiment_path, experiment_name,
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

    def estimate_nb_batches_per_epoch(self):
        logging.info("Learn2track: Estimating training epoch statistics...")
        n_train_batches_capped, _ = self._estimate_nb_batches_per_epoch(
            self.train_batch_sampler, self.train_batch_loader)

        n_valid_batches_capped = None
        if self.valid_batch_sampler is not None:
            logging.info("Learn2track: Estimating validation epoch "
                         "statistics...")
            n_valid_batches_capped, _ = self._estimate_nb_batches_per_epoch(
                self.valid_batch_sampler, self.valid_batch_loader)

        return n_train_batches_capped, n_valid_batches_capped

    # train_validate_and_save_loss  as super
    # train_one_epoch               as super
    # validate_one_epoch            as super

    @classmethod
    def init_from_checkpoint(
            cls, train_batch_sampler: DWIMLBatchSampler,
            valid_batch_sampler: DWIMLBatchSampler,
            train_batch_loader: BatchLoaderOneInput,
            valid_batch_loader: BatchLoaderOneInput,
            model: Learn2TrackModel,
            checkpoint_state: dict, new_patience, new_max_epochs):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).
        """

        # Use super's method but return this learn2track trainer as 'cls'.
        experiment = super(cls, cls).init_from_checkpoint(
            train_batch_sampler, valid_batch_sampler,
            train_batch_loader, valid_batch_loader, model,
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

    def _estimate_nb_batches_per_epoch(self, batch_sampler: DWIMLBatchSampler,
                                       batch_loader: BatchLoaderOneInput):
        """
        Compute the number of batches necessary to use all the available data
        for an epoch (but limiting this to max_nb_batches).

        Returns
        -------
        n_batches : int
            Approximate number of updates per epoch
        batch_size : int
            Batch size or approximate batch size.
        """
        # Here, 'batch_size' will be computed in terms of number of
        # streamlines.
        dataset_size = batch_sampler.dataset.total_nb_streamlines[
            batch_sampler.streamline_group_idx]

        if batch_sampler.batch_size_units == 'nb_streamlines':
            # Then the batch size may actually be different, if some
            # streamlines were split during data augmentation. But still, to
            # use all the data in one epoch, we simply need to devide the
            # dataset_size by this:
            batch_size = batch_sampler.batch_size
        else:  # batch_sampler.batch_size_units == 'length_mm':
            # Then the batch size is more or less exact (with the added
            # gaussian noise possibly changing this a little bit but not much).
            # But we don't know the actual size in number of streamlines.
            batch_size, _ = self._compute_stats_on_a_few_batches(batch_sampler,
                                                                 batch_loader)

        # toDo
        #  None of these cases ensure us a fixed number of input points. If
        #  streamlines are compressed, there isn't much more we can do. If
        #  streamlines have been resampled during loading, though, we can
        #  approximate the actual batch size in number of points after data
        #  augmentation (ignored 2nd returned arg above). But how would we know
        #  how many batches are needed per epoch?

        # Define the number of batches per epoch
        n_batches = int(dataset_size / batch_size)
        n_batches_capped = min(n_batches, self.max_batches_per_epochs)

        logging.info("Dataset had {} streamlines (before data augmentation) "
                     "and each batch contains ~{} streamlines.\nWe will be "
                     "using approximately {} batches per epoch (but not more "
                     "than the allowed {}).\n"
                     .format(dataset_size, batch_size, n_batches,
                             self.max_batches_per_epochs))

        return n_batches_capped, batch_size

    @staticmethod
    def _compute_stats_on_a_few_batches(batch_sampler, batch_loader):
        """
        Since the exact data weight per batch can vary based on data
        augmentation in the batch sampler, we approximate the epoch stats
        using a sample batch.
        """
        # Use temporary RNG states to preserve random "coherency"
        # e.g. when resuming an experiment
        sampler_rng_state_bk = batch_sampler.np_rng.get_state()

        dataloader = DataLoader(batch_sampler.dataset,
                                batch_sampler=batch_sampler,
                                num_workers=0,
                                collate_fn=batch_loader.load_batch)

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
        logging.info("Batch sampler has been ")
        batches_nb_points = []
        batches_nb_streamlines = []
        for sample_data in sample_batches:
            batches_nb_streamlines.append(len(sample_data))
            batches_nb_points.append(sum([len(s) for s in sample_data]))

        avg_batch_size_nb_streamlines = int(np.mean(batches_nb_streamlines))
        avg_batch_size_nb_points = int(np.mean(batches_nb_points))
        if avg_batch_size_nb_points == 0:
            raise ValueError("The allowed batch size ({}) is too small! "
                             "Sampling 0 streamlines per batch."
                             .format(batch_sampler.batch_size))

        logging.info("We have computed that in average, each batch has a "
                     "size of ~{} streamlines for a total of ~{} number of "
                     "datapoints)"
                     .format(avg_batch_size_nb_streamlines,
                             avg_batch_size_nb_points))

        return avg_batch_size_nb_streamlines, avg_batch_size_nb_points

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
