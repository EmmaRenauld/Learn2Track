# -*- coding: utf-8 -*-
"""
learn2track_lstm.py
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca)
date: 23/08/2018

Main code for the Learn2Track project. Learn to reproduce tractography
streamlines in diffusion MRI volumes using a recurrent network.
"""
import hashlib
import importlib.util
import pathlib
import random

from VITALabAI.project.learn2track.utils.errors import EarlyStoppingError

USE_COMET = False
if importlib.util.find_spec("comet_ml"):
    from comet_ml import Experiment as CometExperiment, ExistingExperiment

    USE_COMET = True

import argparse
import contextlib
import json
import logging
import os
import shutil
import time
from argparse import RawTextHelpFormatter
from os.path import join as pjoin
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from torch import optim
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data.dataloader import DataLoader

from VITALabAI import VITALabAiAbstract
from VITALabAI.dataset.learn2track.tractography_recurrent_dataset_done import (LazyLearn2TrackMultiSubjectDataset,
                                                                               Learn2TrackMultiSubjectDataset,
                                                                               Learn2TrackTimestepsBatchSampler)
from VITALabAI.dataset.learn2track.utils import Timer
from VITALabAI.model.generative.learn2track.output_model import (CosineRegressionOutput, FisherVonMisesOutput,
                                                                 GaussianMixtureOutput, GaussianOutput,
                                                                 L2RegressionOutput, SphereClassificationOutput)
from VITALabAI.model.generative.learn2track.rnn_tracking import RecurrentTrackingModel
from VITALabAI.model.generative.learn2track.stacked_rnn import RNN_KEY_TO_CLASS, StackedRNN
from VITALabAI.project.learn2track.utils.debug import log_gpu_memory_usage
from VITALabAI.project.learn2track.utils.experiment_utils import (EarlyStopping, IterTimer, ValueMonitor,
                                                                  compute_gradient_norm)
from dwi_ml.training.trainer_abstract import (DWIMLTrainer)
from dwi_ml.model.batch_samplers import BatchSequencesSamplerOneInputVolume
from dwi_ml.experiment.monitoring import ValueHistoryMonitor


MAX_EPOCH_LENGTH = 10000
OUTPUT_KEY_TO_CLASS = {'cosine-regression': CosineRegressionOutput,
                       'l2-regression': L2RegressionOutput,
                       'sphere-classification': SphereClassificationOutput,
                       'gaussian': GaussianOutput,
                       'gaussian-mixture': GaussianMixtureOutput,
                       'fisher-von-mises': FisherVonMisesOutput}


class Learn2TrackRNNExperiment(VITALabAiAbstract):
    """ Base RNN model for Learn2Track """

    def __init__(self, rnn_model: str, output_model: str, train_database_path: str,
                 valid_database_path: str, name: str = None,
                 batch_size: int = 20000,
                 volumes_per_batch: int = None, n_epoch: int = 100,
                 seed: int = None, streamline_noise_sigma: bool = False,
                 streamlines_cut_ratio: float = None,
                 step_size: float = None, layers: List[int] = None,
                 dropout: float = 0., skip_connections: bool = False,
                 layer_normalization: bool = False,
                 add_neighborhood: float = None,
                 add_previous_dir: bool = False, clip_grad: float = None,
                 weight_decay: float = 0.,
                 patience: int = 20, use_gpu: bool = True, num_workers: int = 0,
                 cycles_per_volume_batch: int = 1,
                 worker_interpolation: bool = False,
                 cache_manager: bool = False,
                 lazy: bool = False, taskman_managed: bool = False):
        """
        Parameters
        ----------
        rnn_model : str
            RNN model to use;
            Choices available are : [lstm | gru]
        output_model : str
            Output model for tracking;
            Choices available are : [regression | sphere-classification | gaussian-mixture]
        train_database_path : str
            Path to training database (hdf5 file)
        valid_database_path : str
            Path to validation database (hdf5 file)
        name : str
            Optional name of the experiment. If given, it is prepended to the auto-generated name.
        batch_size : int
            Number of time steps to use in a batch (the length of sequences vary a lot,
            so we define the number of time steps to use a more consistent amount of memory)
        volumes_per_batch : int
            Limit the number of sampled volumes inside a single batch.
            If None, use true random sampling.
        n_epoch : int
            Maximum number of epochs
        seed : int
            Seed for random numbers
        streamline_noise_sigma : bool
            Add random gaussian noise to streamline coordinates on-the-fly.
            Noise variance is 0.1 * step-size, or 0.1mm if no step size is used.
        streamlines_cut_ratio : float
            Percentage of streamlines to randomly cut in each batch. I
            f None, do not split streamlines.
                NOTE: Preprocessed .hdf5 file should contain resampled streamlines;
                otherwise, cutting streamlines will be biased towards long segments (less points)
        step_size : float
            Constant step size that every streamline should have between points (in mm).
            If None, train on streamlines as they are (e.g. compressed).
        layers : list of int
            List of hidden layers sizes
        dropout : float
            Dropout probability to use in LSTM hidden layers.
        skip_connections : bool
            Add skip connections from the input to all hidden layers, and from
            all hidden layers to the output layer.
        layer_normalization : bool
            Apply layer normalization after each RNN layer.
        add_neighborhood : float
            If given, add neighboring information to the input signal_done at the
            given distance in each axis (in mm).
        add_previous_dir : bool
            If given, add the previous streamline direction to the input signal_done.
        clip_grad : float
            Clip the gradient norm of the model's parameters to address the
            exploding gradient problem in RNNs
        weight_decay : float
            Add a weight decay penalty on the parameters
        patience : int
            Use early stopping. Defines the number of epochs after which
            the model should stop training if the loss hasn't improved. (default: 20)
        use_gpu : bool
            Use the GPU; if False, use CPU
        num_workers : int
            Number of processes that should process the data_DONE between training updates
        cycles_per_volume_batch : int
            Number of batches where the same volumes will be reused before sampling new volumes
        worker_interpolation : bool
            If True and num_workers > 0, interpolation will be done on CPU by the workers.
            Otherwise, interpolation is done on the main thread using the chosen device.
        cache_manager : bool
            If True, use a cache manager to keep volumes and streamlines in memory
        lazy : bool
            If True, use a lazy dataset
        taskman_managed : bool
            If True, taskman manages the experiment. Do not output progress
            bars and instead output special messages for taskman.
        """
        args = {key: value for key, value in locals().items() if key not in ['__builtins__', 'self']}
        print("Running experiment using args: {}".format(json.dumps(args, indent=4, sort_keys=True)))

        # Init properties
        self.rnn_model = rnn_model.lower()
        self.output_model = output_model.lower()
        self.train_database_path = train_database_path
        self.valid_database_path = valid_database_path
        self.name = name
        self.batch_size = int(batch_size)
        self.volumes_per_batch = volumes_per_batch
        self.n_epoch = int(n_epoch)
        self.seed = seed if seed else int(random.random() * 100000)
        self.streamline_noise_sigma = streamline_noise_sigma
        self.streamlines_cut_ratio = streamlines_cut_ratio
        self.step_size = step_size
        self.layers = layers if layers else [100]
        self.dropout = dropout
        self.skip_connections = skip_connections
        self.layer_normalization = layer_normalization
        self.add_neighborhood = add_neighborhood
        self.add_previous_dir = add_previous_dir
        self.clip_grad = clip_grad
        self.weight_decay = weight_decay
        self.patience = patience
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.worker_interpolation = worker_interpolation
        self.cycles_per_volume_batch = cycles_per_volume_batch
        self.cache_manager = cache_manager
        self.lazy = lazy
        self.taskman_managed = taskman_managed

        if not self.name:
            self.name = self._get_experiment_hash()

        self.taskman_report = {
            'loss_train': None,
            'loss_valid': None,
            'epoch': None,
            'best_epoch': None,
            'best_loss': None,
            'update': None,
            'update_loss': None
        }

        # Time limited run
        self.hangup_time = None
        htime = os.environ.get('HANGUP_TIME', None)
        if htime is not None:
            self.hangup_time = int(htime)
            print('Will hang up at ' + htime)

        # Set device
        self.device = None
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Set random numbers
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)  # Set torch seed
        if self.use_gpu:
            torch.cuda.manual_seed(self.seed)

        # Init datasets
        other_kw_args = {}
        if self.lazy:
            dataset_cls = LazyLearn2TrackMultiSubjectDataset
            if self.cache_manager:
                other_kw_args['cache_size'] = self.volumes_per_batch
        else:
            dataset_cls = Learn2TrackMultiSubjectDataset

        # If using worker_interpolation, data_DONE is processed on CPU
        dataset_device = torch.device('cpu') if self.worker_interpolation else self.device

        self.train_dataset = dataset_cls(self.train_database_path, self.rng,
                                         streamline_noise_sigma=self.streamline_noise_sigma,
                                         step_size=self.step_size,
                                         add_neighborhood=self.add_neighborhood,
                                         streamlines_cut_ratio=self.streamlines_cut_ratio,
                                         add_previous_dir=self.add_previous_dir,
                                         do_interpolation=self.worker_interpolation,
                                         device=dataset_device,
                                         taskman_managed=self.taskman_managed, **other_kw_args)
        self.valid_dataset = dataset_cls(self.valid_database_path, self.rng,
                                         streamline_noise_sigma=False,
                                         step_size=self.step_size,
                                         add_neighborhood=self.add_neighborhood,
                                         streamlines_cut_ratio=None,
                                         add_previous_dir=self.add_previous_dir,
                                         do_interpolation=self.worker_interpolation,
                                         device=dataset_device,
                                         taskman_managed=self.taskman_managed, **other_kw_args)

        # Init tasks
        self.early_stopping = EarlyStopping(patience=self.patience)

        # Other variables
        self.input_size = None  # Will be set once the dataset is loaded

        self.model = None
        self.best_model_state = None
        self.best_epoch = None

        self.optimizer = None

        self.current_epoch = 0
        self.experiment_dir = self.name

        # Setup monitors
        self.train_loss_monitor = ValueMonitor("Training loss")
        self.valid_loss_monitor = ValueMonitor("Validation loss")
        self.grad_norm_monitor = ValueMonitor("Grad Norm")

        # Comet
        self.comet_exp = None
        self.comet_key = None

    def load_dataset(self):
        """
        This method loads the data_DONE (streamlines and data_DONE volume).
        """
        with Timer("Loading training dataset", newline=True, color='blue'):
            self.train_dataset.load()

            model_input_size = self._compute_input_size()

            self.input_size = model_input_size

        with Timer("Loading validation dataset", newline=True, color='blue'):
            self.valid_dataset.load()

    def _compute_input_size(self):
        expected_input_size = self.train_dataset.tractodata_manager.feature_size
        if self.add_neighborhood:
            expected_input_size += 26 * self.train_dataset.tractodata_manager.feature_size
        if self.add_previous_dir:
            expected_input_size += 128
        return expected_input_size

    def _should_quit(self, iter_timer):
        # If:
        #   hang up signal_done received
        #   time remaining is less than one epoch + 30 seconds
        # exit training.
        return self.hangup_time is not None and time.time() + iter_timer.mean * 2.0 + 30 > self.hangup_time

    def _update_taskman_report(self, updates):
        self.taskman_report.update(updates)
        self.taskman_report['time'] = time.time()
        print('!taskman' + json.dumps(self.taskman_report), flush=True)

    def build_model(self):
        """
        Build PyTorch LSTM model
        """
        # `input_size` is inferred from the dataset
        if self.input_size is None:
            self.load_dataset()

        output_model_cls = OUTPUT_KEY_TO_CLASS[self.output_model]  # BaseTrackingOutputModel

        # Assert that the output model supports compressed streamlines
        if self.step_size is None and not output_model_cls.supportsCompressedStreamlines:
            raise ValueError("Output model '{}' does not support compressed streamlines."
                             "Please specifiy a step size.".format(self.output_model))

        stacked_rnn = StackedRNN(rnn_base=self.rnn_model,
                                 input_size=self.input_size,
                                 layer_sizes=self.layers,
                                 skip_connections=self.skip_connections,
                                 layer_normalization=self.layer_normalization,
                                 dropout=self.dropout)
        output_model = output_model_cls(hidden_size=stacked_rnn.output_size,
                                        dropout=self.dropout)

        self.model = RecurrentTrackingModel(stacked_rnn, output_model, self.add_previous_dir)

        # Send model to device
        # NOTE: This ordering is important! The optimizer needs to use the cuda Tensors if using the GPU...
        self.model.to(device=self.device)

        # Build optimizer (Optimizer is built here since it needs the model parameters)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=self.weight_decay)

        # Initialize best model
        self.best_model_state = self.model.state_dict()

    def train(self):
        """
        Train the model
        """
        # An API key should be set in $HOME/.comet.config with contents:
        # [comet]
        # api_key=YOUR-API-KEY
        self._init_comet()

        print("Experiment attributes :")
        print(json.dumps(self.attributes, indent=4, sort_keys=True, default=(lambda x: str(x))))
        print("Experiment hyperparameters :")
        print(json.dumps(self.hyperparameters, indent=4, sort_keys=True, default=(lambda x: str(x))))

        # Initialize dataloaders
        train_sampler = Learn2TrackTimestepsBatchSampler(data_source=self.train_dataset,
                                                         batch_size=self.batch_size,
                                                         rng=self.rng,
                                                         n_volumes=self.volumes_per_batch,
                                                         cycles=self.cycles_per_volume_batch)
        valid_sampler = Learn2TrackTimestepsBatchSampler(data_source=self.valid_dataset,
                                                         batch_size=self.batch_size,
                                                         rng=self.rng,
                                                         n_volumes=self.volumes_per_batch,
                                                         cycles=self.cycles_per_volume_batch)

        # Pin memory if interpolation is done by workers; this means that
        # dataloader output is on GPU, ready to be fed to the model.
        # Otherwise, dataloader output is kept on CPU, and the main thread sends
        # volumes and coords on GPU for interpolation.
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_sampler=train_sampler,
                                      num_workers=self.num_workers,
                                      collate_fn=self.train_dataset.collate_fn,
                                      pin_memory=self.use_gpu and self.worker_interpolation)
        valid_dataloader = DataLoader(self.valid_dataset, batch_sampler=valid_sampler,
                                      num_workers=self.num_workers,
                                      collate_fn=self.valid_dataset.collate_fn,
                                      pin_memory=self.use_gpu and self.worker_interpolation)

        print("Estimating training epoch statistics...")
        train_epoch_length, train_sequence_size = self._compute_epoch_stats(self.train_dataset, self.batch_size)
        n_train_batches_capped = min(train_epoch_length, MAX_EPOCH_LENGTH)
        print("Training dataset has {} streamlines. \n"
              "Training with approximately {} iterations per epoch (out of {} uncapped) "
              "using approximately {} sequences per batch "
              "({} timesteps per batch)\n".format(len(self.train_dataset),
                                                  n_train_batches_capped,
                                                  train_epoch_length,
                                                  train_sequence_size,
                                                  self.batch_size))

        print("Estimating validation epoch statistics...")
        valid_epoch_length, valid_sequence_size = self._compute_epoch_stats(self.valid_dataset, self.batch_size)
        n_valid_batches_capped = min(valid_epoch_length, MAX_EPOCH_LENGTH)
        print("Validation dataset has {} streamlines. \n"
              "Validating with approximately {} iterations per epoch (out of {} uncapped) "
              "using approximately {} sequences per batch "
              "({} timesteps per batch)\n".format(len(self.valid_dataset),
                                                  n_valid_batches_capped,
                                                  valid_epoch_length,
                                                  valid_sequence_size,
                                                  self.batch_size))

        iter_timer = IterTimer(history_len=20)

        train_context = contextlib.nullcontext
        valid_context = contextlib.nullcontext
        if self.comet_exp:
            train_context = self.comet_exp.train
            valid_context = self.comet_exp.validate

        # Start from current_spoch in case the experiment is resuming
        for epoch in iter_timer(range(self.current_epoch, self.n_epoch)):
            self.current_epoch = epoch
            print("Epoch #{}".format(epoch))
            if self.comet_exp:
                self.comet_exp.log_metric("current_epoch", self.current_epoch)

            # Make sure there are no existing HDF handles if using parallel workers
            if self.num_workers > 0 and isinstance(self.train_dataset, LazyLearn2TrackMultiSubjectDataset):
                self.train_dataset.hdf_handle = None
                self.train_dataset.volume_cache_manager = None
                self.valid_dataset.hdf_handle = None
                self.valid_dataset.volume_cache_manager = None

            with tqdm.tqdm(train_dataloader, ncols=100, disable=self.taskman_managed,
                           total=n_train_batches_capped) as pbar:
                train_iterator = enumerate(pbar)
                with train_context():
                    for step, data in train_iterator:
                        # Break if maximum number of epochs has been reached
                        if step == n_train_batches_capped:
                            # Explicitly close tqdm's progress bar to fix possible bugs when breaking the loop
                            pbar.close()
                            break

                        mean_loss = self._run_model(data, is_training=True)
                        self.train_loss_monitor.update(mean_loss)
                        logging.debug("Update loss: {}".format(mean_loss))

                        # Update taskman every 10 updates
                        if self.taskman_managed and step % 10 == 0:
                            updates = {
                                'loss_train': self.train_loss_monitor.epochs[-1] if len(
                                    self.train_loss_monitor.epochs) > 0 else None,
                                'loss_valid': self.valid_loss_monitor.epochs[-1] if len(
                                    self.valid_loss_monitor.epochs) > 0 else None,
                                'epoch': self.current_epoch,
                                'best_epoch': self.best_epoch,
                                'best_loss': self.early_stopping.best,
                                'update': step,
                                'update_loss': mean_loss
                            }
                            self._update_taskman_report(updates)

                        # Update Comet every 10 updates
                        if self.comet_exp and step % 10 == 0:
                            self.comet_exp.log_metric("loss_step",
                                                      mean_loss,
                                                      step=step)
                            self.comet_exp.log_metric("gradient_norm_step",
                                                      self.grad_norm_monitor.history[-1],
                                                      step=step)

            self.train_loss_monitor.end_epoch()
            self.grad_norm_monitor.end_epoch()

            self._save_log_array(self.train_loss_monitor.epochs, "train_loss.npy")
            self._save_log_array(self.grad_norm_monitor.epochs, "gradient_norm.npy")

            with train_context():
                if self.comet_exp:
                    self.comet_exp.log_metric("loss_epoch",
                                              self.train_loss_monitor.epochs[-1],
                                              step=epoch)
                    self.comet_exp.log_metric("gradient_norm_epoch",
                                              self.grad_norm_monitor.epochs[-1],
                                              step=epoch)

            print("Mean training loss : {}".format(self.train_loss_monitor.epochs[-1]))
            print("Mean gradient norm : {}".format(self.grad_norm_monitor.epochs[-1]))

            # Explicitly delete iterator to kill threads and free memory before running validation
            del train_iterator

            # Validation
            with tqdm.tqdm(valid_dataloader, ncols=100,
                           disable=self.taskman_managed,
                           total=n_valid_batches_capped) as pbar:
                valid_iterator = enumerate(pbar)
                for step, data in valid_iterator:
                    # Break if maximum number of epochs has been reached
                    if step == n_valid_batches_capped:
                        # Explicitly close tqdm's progress bar to fix possible bugs when breaking the loop
                        pbar.close()
                        break
                    mean_loss = self._run_model(data)
                    self.valid_loss_monitor.update(mean_loss)

            self.valid_loss_monitor.end_epoch()
            self._save_log_array(self.valid_loss_monitor.epochs, "valid_loss.npy")

            with valid_context():
                if self.comet_exp:
                    self.comet_exp.log_metric("loss_epoch",
                                              self.valid_loss_monitor.epochs[-1],
                                              step=epoch)

            print("Validation loss : {}".format(self.valid_loss_monitor.epochs[-1]))

            # Explicitly delete iterator to kill threads and free memory before running training again
            del valid_iterator

            # Check for early stopping
            if self.early_stopping.step(self.valid_loss_monitor.epochs[-1]):
                msg = "Early stopping! Loss has not improved after {} epochs...".format(self.patience)
                msg += "\nBest result: {}; At epoch #{}".format(self.early_stopping.best, self.best_epoch)
                self.save_checkpoint()
                raise EarlyStoppingError(msg)

            # Check for current best
            if self.valid_loss_monitor.epochs[-1] < (self.early_stopping.best + self.early_stopping.min_eps):
                print("Best epoch yet!")
                self.best_model_state = self.model.state_dict()
                self.best_epoch = self.current_epoch
                self.save_model()

                if self.comet_exp:
                    self.comet_exp.log_metric("best_validation", self.early_stopping.best)
                    self.comet_exp.log_metric("best_epoch", self.best_epoch)

            # End of epoch, save checkpoint for resuming later
            self.save_checkpoint()

            if self.taskman_managed:
                updates = {
                    'loss_train': self.train_loss_monitor.epochs[-1],
                    'loss_valid': self.valid_loss_monitor.epochs[-1],
                    'epoch': self.current_epoch,
                    'best_epoch': self.best_epoch,
                    'best_loss': self.early_stopping.best
                }
                self._update_taskman_report(updates)

            # (For taskman) Check if time is running out
            if self._should_quit(iter_timer):
                print('Seems like I should quit, so I quit.')
                print('Remaining: {:.0f} s'.format(self.hangup_time - time.time()))
                self._update_taskman_report({'resubmit': True})
                exit(2)

        # Training is over, save checkpoint
        self.save_checkpoint()

    def _run_model(self, data, is_training: bool = False) -> float:
        """Run a batch of data_DONE through the model and return the mean loss.

        Parameters
        ----------
        data : tuple of (List, dict)
            The streamlines and a dictionary mapping each tractodata_id to a slice of streamlines
        is_training : bool
            If True, record the computation graph and backprop through the model
            parameters.

        Returns
        -------
        mean_loss : float
            The mean loss of the provided batch
        """
        if is_training:
            # If training, enable gradients for backprop
            self.model.train()
            grad_context = torch.enable_grad
        else:
            # If evaluating, turn gradients off
            self.model.eval()
            grad_context = torch.no_grad

        with grad_context():
            # GPU interpolation needs to be done here in the main thread
            if not self.worker_interpolation:
                voxel_streamlines, batch_tid_to_slice = data
                data = self.train_dataset.get_model_inputs_from_streamlines(voxel_streamlines,
                                                                            batch_tid_to_slice,
                                                                            device=self.device)

            packed_inputs, packed_targets = data
            if self.use_gpu:
                packed_inputs = packed_inputs.cuda()
                packed_targets = packed_targets.cuda()

            logging.debug("Running model on a batch of {} streamlines".format(packed_inputs.batch_sizes[0]))

            if is_training:
                # Reset parameter gradients
                self.optimizer.zero_grad()

            try:
                # Apply model
                model_outputs, _ = self.model(packed_inputs)
            except RuntimeError:
                # Training RNNs with variable-length sequences on the GPU can cause
                # memory fragmentation in the pytorch-managed cache, possibly
                # leading to "random" OOM RuntimeError during training.
                # Emptying the GPU cache seems to fix the problem for now.
                # We don't do it every update because it can be time consuming.
                torch.cuda.empty_cache()
                model_outputs, _ = self.model(packed_inputs)

            # Compute loss
            mean_loss = self.model.compute_loss(model_outputs, packed_targets)

            if is_training:
                # Backprop loss
                mean_loss.backward()

                # Clip gradient if necessary before updating parameters
                if self.clip_grad:
                    total_norm = compute_gradient_norm(self.model.parameters())
                    self.grad_norm_monitor.update(total_norm)
                    logging.debug("Gradient norm: {}".format(total_norm))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

                # Update parameters
                self.optimizer.step()

            if self.use_gpu:
                log_gpu_memory_usage()

        return mean_loss.cpu().item()

    @property
    def attributes(self):
        """Return experiment attributes (anything that is not a hyperparameter).
        """
        attrs = {
            'name': self.name,
            'version': 8,
            'train_database_path': self.train_database_path,
            'valid_database_path': self.valid_database_path,
            'input_size': self.input_size,
            'comet_key': self.comet_key
        }
        return attrs

    @property
    def hyperparameters(self):
        """
        Return experiment hyperparameters in a dictionary
        """
        hyperparameters = super().hyperparameters
        hyperparameters.update({
            'model_attributes': self.model.attributes,
        })
        return hyperparameters

    def save_model(self):
        """
        This method is called at the end to save the best parameters for the model.
        """
        # Make model directory
        model_dir = pjoin(self.experiment_dir, "model")

        # If a model was already saved, back it up and erase it after saving the new
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = pjoin(self.experiment_dir, "model_old")
            shutil.move(model_dir, to_remove)

        os.makedirs(model_dir)

        # Save attributes
        with open(pjoin(model_dir, "attributes.json"), 'w') as json_file:
            json_file.write(json.dumps(self.attributes, indent=4, separators=(',', ': ')))

        # Save hyperparams
        with open(pjoin(model_dir, "hyperparameters.json"), 'w') as json_file:
            json_file.write(json.dumps(self.hyperparameters, indent=4, separators=(',', ': ')))

        # Save losses
        losses = {'train_loss': self.train_loss_monitor.epochs[self.best_epoch],
                  'valid_loss': self.early_stopping.best}
        with open(pjoin(model_dir, "losses.json"), 'w') as json_file:
            json_file.write(json.dumps(losses, indent=4, separators=(',', ': ')))

        # Save model
        torch.save(self.best_model_state, pjoin(model_dir, "best_model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    def save_checkpoint(self):
        """
        Save an experiment checkpoint that can be resumed from.
        """

        # Make model directory
        checkpoint_dir = pjoin(self.experiment_dir, "checkpoint")

        # Backup old checkpoint before saving, and erase it afterwards
        to_remove = None
        if os.path.exists(checkpoint_dir):
            to_remove = pjoin(self.experiment_dir, "checkpoint_old")
            shutil.move(checkpoint_dir, to_remove)

        os.makedirs(checkpoint_dir)

        # Save experiment
        checkpoint_state = {
            'epoch': self.current_epoch,
            'best_epoch': self.best_epoch,
            'attributes': self.attributes,
            'hyperparameters': self.hyperparameters,
            'optimizer': self.optimizer.state_dict(),
            'weights': self.model.state_dict(),
            'torch_rng_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state() if self.use_gpu else None,
            'numpy_rng_state': self.rng.get_state(),
            'early_stopping': self.early_stopping.get_state(),
            'train_loss_monitor': self.train_loss_monitor.get_state(),
            'valid_loss_monitor': self.valid_loss_monitor.get_state(),
            'grad_norm_monitor': self.grad_norm_monitor.get_state()
        }
        torch.save(checkpoint_state, pjoin(checkpoint_dir, "checkpoint_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    @classmethod
    def load_checkpoint(cls, path: str, **kwargs: Dict[str, Any]):
        """
        Load a saved checkpoint, and return a new experiment.

        Parameters
        ----------
        path : str
            Path to experiment folder
        kwargs : dict
            Arguments that will override the loaded hyperparameters
        """
        checkpoint_state = torch.load(pjoin(path, "checkpoint", "checkpoint_state.pkl"))
        experiment_params = checkpoint_state['hyperparameters'].copy()
        experiment_params.update(checkpoint_state['attributes'])
        experiment_params.update(kwargs)

        # Retrocompatibility for older versions
        if experiment_params['version'] < 2:
            experiment_params['model_key'] = 'LSTM'

        if experiment_params['version'] >= 3:
            if "input_size" in experiment_params:
                del experiment_params['input_size']
            if "sh_order" in experiment_params:
                del experiment_params['sh_order']

            if "add_previous_direction" in experiment_params:
                if experiment_params["add_previous_direction"]:
                    raise ValueError("Support for option --add-previous-direction was removed")
                del experiment_params["add_previous_direction"]

        if experiment_params['version'] < 6:
            # Switched from a single model_key to a rnn-model/output-model combination
            model_key_to_rnn_output = {'lstm-det': ['lstm', 'regression'],
                                       'lstm-prob': ['lstm', 'sphere-classification'],
                                       'lstm-prob-mixt': ['lstm', 'gaussian-mixture'],
                                       'gru-det': ['gru', 'regression'],
                                       'gru-prob': ['gru', 'sphere-classification'],
                                       'gru-prob-mixt': ['gru', 'gaussian-mixture']
                                       }
            rnn_model, output_model = model_key_to_rnn_output[experiment_params['model_key']]
            experiment_params['rnn_model'] = rnn_model
            experiment_params['output_model'] = output_model
            del experiment_params['model_key']

        if experiment_params['version'] < 8:
            experiment_params['skip_connections'] = False
            experiment_params['layer_normalization'] = False
            experiment_params['layers'] = [experiment_params['hidden_size']] * experiment_params['num_layers']
            del experiment_params['hidden_size']
            del experiment_params['num_layers']

        # Comet.ml support
        if "comet_key" in experiment_params:
            del experiment_params['comet_key']
        del experiment_params['version']

        # Get a new experiment
        experiment = cls(**experiment_params)
        experiment.experiment_dir = path

        # Set RNG states
        torch.set_rng_state(checkpoint_state['torch_rng_state'])
        experiment.rng.set_state(checkpoint_state['numpy_rng_state'])
        if experiment.use_gpu and checkpoint_state['torch_cuda_state'] is not None:
            torch.cuda.set_rng_state(checkpoint_state['torch_cuda_state'])

        # Set other objects
        experiment.current_epoch = checkpoint_state['epoch'] + 1
        experiment.best_epoch = checkpoint_state['best_epoch']
        experiment.early_stopping.set_state(checkpoint_state['early_stopping'])
        experiment.early_stopping.patience = experiment.patience  # Patience overloading
        experiment.train_loss_monitor.set_state(checkpoint_state['train_loss_monitor'])
        experiment.valid_loss_monitor.set_state(checkpoint_state['valid_loss_monitor'])
        experiment.grad_norm_monitor.set_state(checkpoint_state['grad_norm_monitor'])
        experiment.comet_key = checkpoint_state['attributes']['comet_key']

        # Stop now if early stopping was triggered so we don't load dataset
        # and build model for no reason
        if experiment.early_stopping.n_bad_epochs >= experiment.early_stopping.patience:
            raise EarlyStoppingError("Resumed experiment was stopped because of early stopping, "
                                     "increase patience in order to resume training!")

        # Load dataset, necessary to get input_size before build_model()
        experiment.load_dataset()

        # Load model
        experiment.build_model()
        experiment.model.load_state_dict(checkpoint_state['weights'])
        experiment.optimizer.load_state_dict(checkpoint_state['optimizer'])

        # Return a new instance to keep a coherent state
        return experiment

    def _compute_epoch_stats(self, dataset: Learn2TrackMultiSubjectDataset, batch_size: int):
        """Compute approximated statistics about epochs.

        Since the exact number of sequences and timesteps per batch can vary,
        we approximate the epoch stats using a sample batch.

        Parameters
        ----------
        dataset : Learn2TrackMultiSubjectDataset
            Dataset to get statistics for.
        batch_size : int
            Batch size required for this dataset

        Returns
        -------
        approx_epoch_length : int
            Approximate number of updates per epoch
        batch_sequence_size : int
            Approximate number of sequences per batch
        """
        # Use temporary RNG states to preserve random "coherency"
        # e.g. when resuming an experiment
        sampler_rng_state_bk = self.rng.get_state()
        dataset_rng_state_bk = dataset.rng.get_state()

        # Backup properties that need to be changed
        streamlines_cut_ratio_bk = dataset.streamlines_cut_ratio

        # Turn off cutting to avoid a virtual increase of the number of
        # sequences per batch. This does not change the number of timesteps
        # used in practice because both segments are kept in the batch when cutting.
        dataset.streamlines_cut_ratio = None

        sampler = Learn2TrackTimestepsBatchSampler(data_source=dataset,
                                                   batch_size=batch_size,
                                                   rng=self.rng,
                                                   n_volumes=self.volumes_per_batch,
                                                   cycles=self.cycles_per_volume_batch)
        dataloader = DataLoader(dataset,
                                batch_sampler=sampler,
                                num_workers=0,
                                collate_fn=dataset.collate_fn)

        # Get a sample batch to compute stats
        sample_batches = [next(iter(dataloader))[0] for i in range(5)]

        # Restore RNG states
        self.rng.set_state(sampler_rng_state_bk)
        dataset.rng.set_state(dataset_rng_state_bk)

        # Restore properties
        dataset.streamlines_cut_ratio = streamlines_cut_ratio_bk

        # VERY IMPORTANT: Reset HDF handles
        # Parallel workers each need to initialize independent HDF5 handles
        if isinstance(dataset, LazyLearn2TrackMultiSubjectDataset):
            dataset.hdf_handle = None
            dataset.volume_cache_manager = None

        # Compute stats about epoch
        batch_sizes = []
        for sample_data in sample_batches:
            if isinstance(sample_data, PackedSequence):
                batch_sizes.append(sample_data.batch_sizes[0])
            else:
                batch_sizes.append(len(sample_data))
        avg_batch_size = int(np.mean(batch_sizes))
        dataset_size = len(dataloader.dataset)
        approx_epoch_length = int(dataset_size / avg_batch_size)

        return approx_epoch_length, avg_batch_size

    def _save_log_array(self, array: np.ndarray, fname: str):
        log_dir = pjoin(self.experiment_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fpath = pjoin(log_dir, fname)
        np.save(fpath, array)

    def _get_experiment_hash(self):
        hyperparams = self.hyperparameters.copy()
        str_repr = json.dumps(hyperparams, ensure_ascii=True, sort_keys=True)
        hash_repr = hashlib.sha256(str_repr.encode()).hexdigest()
        return hash_repr

    def _init_comet(self):
        # An API key should be set in $HOME/.comet.config with contents:
        # [comet]
        # api_key=YOUR-API-KEY
        if not USE_COMET:
            return
        try:
            if self.comet_key:
                self.comet_exp = ExistingExperiment(previous_experiment=self.comet_key)
            else:
                # New experiment
                # Use trainset name as comet project name
                project_name = pathlib.Path(self.train_database_path).stem
                self.comet_exp = CometExperiment(project_name=project_name,
                                                 workspace="ppoulin",
                                                 log_code=False,
                                                 log_graph=True,
                                                 auto_param_logging=True,
                                                 auto_metric_logging=False,
                                                 parse_args=False,
                                                 auto_output_logging='native',
                                                 log_env_details=True,
                                                 log_env_gpu=True,
                                                 log_env_cpu=True,
                                                 log_env_host=False,
                                                 log_git_metadata=True,
                                                 log_git_patch=True,
                                                 display_summary=False,
                                                 disabled=False)
                self.comet_exp.set_name(self.name)
                self.comet_exp.log_parameters(self.attributes)
                self.comet_exp.log_parameters(self.hyperparameters)
                self.comet_key = self.comet_exp.get_key()
        except ConnectionError:
            print("Could not connect to Comet.ml, metrics will not be "
                  "logged online...")
            self.comet_exp = None
            self.comet_key = None


def parse_args():
    """Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(description=str(parse_args.__doc__),
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('train_database_path', type=str,
                        help="Path to the training set (.hdf5).")
    parser.add_argument('valid_database_path', type=str,
                        help="Path to the validation set (.hdf5).")
    parser.add_argument('--rnn-model', choices=RNN_KEY_TO_CLASS.keys(),
                        required=True, help="Type of RNN model to use.")
    parser.add_argument('--output-model', choices=OUTPUT_KEY_TO_CLASS.keys(),
                        required=True, help="Type of output to use.")
    parser.add_argument('--name', type=str, help="Name of the experiment.")
    parser.add_argument('--n-epoch', type=int, default=100,
                        help="Maximum number of epochs.")
    parser.add_argument('--batch-size', type=int, default=20000,
                        help="Number of streamline points per batch.")
    parser.add_argument('--seed', type=int,
                        help="Random experiment seed.")
    parser.add_argument('--streamline-noise-sigma', action="store_true",
                        help="Add random gaussian noise to streamline coordinates on-the-fly. "
                             "Noise variance is 0.1 * step-size, or 0.1mm if no step size is used.")
    parser.add_argument('--streamlines-cut-ratio', type=float,
                        help="Cut a percentage of streamline at a random point in each batch.")
    parser.add_argument('--step-size', type=float,
                        help="Resample all streamlines to this step size. "
                             "If None, train on streamlines as they are (e.g. compressed).")
    parser.add_argument('--layers', type=int, nargs='+', metavar='layer_size',
                        help="List of the model's hidden layers size.")
    parser.add_argument('--dropout', type=float, default=0.,
                        help="Rate of units that will be dropped after each layer.")
    parser.add_argument('--skip-connections', action="store_true",
                        help="Add skip connections from the input to all hidden "
                             "layers, and from all hidden layers to the output layer.")
    parser.add_argument('--layer-normalization', action="store_true",
                        help="Apply layer normalization after each RNN layer.")
    parser.add_argument('--add-neighborhood', type=float,
                        help="Concatenate interpolated neighborhood information to the input vector.")
    parser.add_argument('--add-previous-dir', action="store_true",
                        help="Concatenate previous streamline direction to the input vector.")
    parser.add_argument('--clip-grad', type=float,
                        help="Clip the gradient norm of the model's parameters "
                             "to address the exploding gradient problem in RNNs")
    parser.add_argument('--patience', type=int, default=20,
                        help="Early-stopping lookahead.")
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help="Weight decay applied on model parameters")
    parser.add_argument('--use-gpu', action="store_true",
                        help="Train using the GPU.")
    parser.add_argument('--num-workers', type=int, default=0,
                        help="Number of parallel CPU workers.")
    parser.add_argument('--lazy', action="store_true",
                        help="Do not load all the training dataset in memory at once."
                             "Load only what is needed for a batch.")
    parser.add_argument('--cache-manager', action="store_true",
                        help="Relevant only if --lazy is used."
                             "Cache volumes and streamlines in-memory instead of fetching from the disk everytime."
                             "Cache size is determined by --volumes-per-batch.")
    parser.add_argument('--volumes-per-batch', type=int,
                        help="Limits the number of volumes used in a batch."
                             "Also determines the cahce size if --cache-manager is used.")
    parser.add_argument('--cycles-per-volume-batch', type=int,
                        help="Relevant only if --volumes-per-batch is used."
                             "Number of update cycles before chaging to new volumes.")
    parser.add_argument('--worker-interpolation', action='store_true',
                        help="If using --num-workers > 0, interpolation will be "
                             "done on CPU by the workers instead of on the main "
                             "thread using the chosen device")
    parser.add_argument('--taskman-managed', action="store_true",
                        help="Instead of printing progression, print taskman-relevant data_DONE.")
    parser.add_argument('--logging', type=str,
                        choices=['error', 'warning', 'info', 'debug'],
                        default='warning',
                        help="Activate debug mode")

    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_args()

    # Initialize logger
    logging.basicConfig(level=str(args.logging).upper())

    experiment_args = vars(args)
    del experiment_args['logging']

    logging.info(args)

    experiment = Learn2TrackRNNExperiment(**experiment_args)

    try:
        if os.path.exists(pjoin(experiment.experiment_dir, "checkpoint")):
            print("Experiment folder exists, resuming experiment!")
            # Resuming on a cluster will change the database path, which is why
            # we allow overwriting the training and validation paths
            args_overwrite = {'train_database_path': experiment.train_database_path,
                              'valid_database_path': experiment.valid_database_path,
                              'n_epoch': experiment.n_epoch,
                              'use_gpu': experiment.use_gpu,
                              'num_workers': experiment.num_workers,
                              'worker_interpolation': experiment.worker_interpolation,
                              'volumes_per_batch': experiment.volumes_per_batch,
                              'cycles_per_volume_batch': experiment.cycles_per_volume_batch,
                              'cache_manager': experiment.cache_manager,
                              'lazy': experiment.lazy,
                              'batch_size': experiment.batch_size,
                              'patience': experiment.patience,
                              'taskman_managed': experiment.taskman_managed}
            # load_checkpoint() already loads dataset and builds model, so do it
            # only if running a new experiment
            experiment = Learn2TrackRNNExperiment.load_checkpoint(experiment.experiment_dir, **args_overwrite)
        else:
            experiment.load_dataset()
            experiment.build_model()

        experiment.train()
    except EarlyStoppingError as e:
        print(e)

    print("Script terminated successfully. Saved experiment in folder : ")
    print(experiment.experiment_dir)


if __name__ == '__main__':
    main()
