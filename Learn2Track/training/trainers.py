# -*- coding: utf-8 -*-
"""
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca),
        refactored by Emmanuelle Renauld
date: 24/08/2021
"""
import logging

import numpy as np
import torch
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_and_normalize_directions
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data.dataloader import DataLoader

from dwi_ml.experiment_utils.learning_utils import compute_gradient_norm
from dwi_ml.experiment_utils.memory import log_gpu_memory_usage
from dwi_ml.data_loaders.batch_samplers import (
    BatchStreamlinesSamplerOneInput as BatchSampler)
from dwi_ml.training.trainers import DWIMLTrainer

from Learn2Track.models.learn2track_model import Learn2TrackModel


class Learn2TrackTrainer(DWIMLTrainer):
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

    def run_one_batch(self, data, is_training: bool,
                      batch_sampler: BatchSampler, *args):
        """Run a batch of data through the model (calling its forward method)
        and return the mean loss. If training, run the backward method too.

        In the trainer, this is called inside the loop:
        for epochs: (goes in train_one_epoch)
            for batches: (goes in train_one_batch)
                self.model.run_model_and_compute_loss

        If the sampler was instantiated with wait_for_gpu, then we need to
        compute the inputs here; not done yet.

        Parameters
        ----------
        data : tuple of (List, dict)
            This is the output of the BatchSequencesSampleOneInputVolume's
            load_batch() function. If wait_for_gpu, data is
            (batch_streamlines, final_streamline_ids_per_subj). Else, data is
            (batch_streamlines, final_streamline_ids_per_subj, inputs)
        batch_sampler: BatchSequencesSamplerOneInputVolume
            Either self.train_batch_sampler or valid_batch_sampler, depending
            on the case.
        is_training : bool
            If True, record the computation graph and backprop through the
            model parameters.
        Returns
        -------
        mean_loss : float
            The mean loss of the provided batch
        total_norm: float
            The total norm (sqrt(sum(params**2))) of parameters before gradient
            clipping, if any.
        """
        if is_training:
            # If training, enable gradients for backpropagation.
            # Uses torch's module train(), which "turns on" the training mode.
            self.model.train()
            grad_context = torch.enable_grad
        else:
            # If evaluating, turn gradients off for back-propagation
            # Uses torch's module eval(), which "turns off" the training mode.
            self.model.eval()
            grad_context = torch.no_grad

        with grad_context():
            if batch_sampler.wait_for_gpu:
                if not self.use_gpu:
                    logging.warning(
                        "Batch sampler has been created with use_gpu=True, so "
                        "some computations have been skipped to allow user to "
                        "compute them later on GPU. Now in training, however, "
                        "you are using CPU, so this was not really useful.\n"
                        "Maybe this is an error in the code?")
                # Data interpolation has not been done yet. GPU computations
                # need to be done here in the main thread. Running final steps
                # of data preparation.
                self.logger.debug('Finalizing input data preparation on GPU.')
                batch_streamlines, final_s_ids_per_subj = data

                # Getting the inputs points from the volumes. Usually done in
                # load_batch but we preferred to wait here to have a chance to
                # run things on GPU.
                batch_inputs = batch_sampler.compute_inputs(
                    batch_streamlines, final_s_ids_per_subj)

            else:
                # Data is already ready
                batch_streamlines, final_s_ids_per_subj, batch_inputs = data

            # Converting streamlines into a list of directions.
            # + computing previous dirs
            # + packing everything
            batch_inputs = self.model.prepare_inputs(batch_inputs)
            unpacked_directions, batch_directions = self.model.prepare_targets(
                batch_streamlines, self.device)
            batch_prev_dirs = self.model.prepare_previous_dirs(
                unpacked_directions, self.device)

            if self.use_gpu:
                # Tensors coming from the Dataloader will be on cpu. Sending
                # to GPU.
                batch_inputs = batch_inputs.cuda()
                batch_directions = batch_directions.cuda()
                if batch_prev_dirs:
                    batch_prev_dirs = batch_prev_dirs.cuda()

            if is_training:
                # Reset parameter gradients
                # See here for some explanation
                # https://stackoverflow.com/questions/48001598/why-do-we-need-
                # to-call-zero-grad-in-pytorch
                self.optimizer.zero_grad()

            self.logger.debug('\n=== Computing forward propagation! ===')
            try:
                # Apply model. This calls our model's forward function
                # (the hidden states are not used here, neither as input nor
                # outputs. We need them only during tracking).
                model_outputs, _ = self.model(batch_inputs, batch_prev_dirs)
            except RuntimeError:
                # Training RNNs with variable-length sequences on the GPU can
                # cause memory fragmentation in the pytorch-managed cache,
                # possibly leading to "random" OOM RuntimeError during
                # training. Emptying the GPU cache seems to fix the problem for
                # now. We don't do it every update because it can be time
                # consuming.
                torch.cuda.empty_cache()
                model_outputs, _ = self.model(batch_inputs, batch_prev_dirs)

            # Compute loss
            self.logger.debug('\n=== Computing loss ===')
            mean_loss = self.model.compute_loss(model_outputs,
                                                batch_directions.data)
            self.logger.info("Loss is : {}".format(mean_loss))

            if is_training:
                self.logger.debug('\n=== Computing back propagation ===')

                # Explanation on the backward here:
                # - Each parameter in the RNN and other sub-networks have been
                #   created with the flag requires_grad=True by torch.
                #   ==> gradients = [i.grad for i in self.model.parameters()]
                # - When using parameters to compute something (ex, outputs)
                #   torch.autograd creates a computational graph, remembering
                #   all the functions that were used from parameters that
                #   contain the requires_grad.
                # - When calling backward, the backward of each sub-function is
                #   called iteratively, each time computing the partial
                #   derivative dloss/dw and modifying the parameters' .grad
                #   ==> model_outputs.grad_fn shows the last used function,
                #       and thus the first backward to be used, here:
                #       MeanBackward0  (last function was a mean)
                #   ==> model_outputs.grad_fn shows that the last used fct
                #       is AddmmBackward  (addmm = matrix multiplication)
                mean_loss.backward()

                # Clip gradient if necessary before updating parameters
                # Remembering unclipped value.
                grad_norm = compute_gradient_norm(self.model.parameters())
                self.logger.debug("Gradient norm: {}".format(grad_norm))
                if self.clip_grad:
                    # self.grad_norm_monitor.update(grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip_grad)
                    grad_norm = compute_gradient_norm(self.model.parameters())
                    self.logger.debug(
                        "Gradient norm when gradients are clipped is {}"
                        .format(grad_norm))

                # Update parameters
                self.optimizer.step()
            else:
                grad_norm = None

            if self.use_gpu:
                log_gpu_memory_usage(self.logger)

        return mean_loss.cpu().item(), grad_norm

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
