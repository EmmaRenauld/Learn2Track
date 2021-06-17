# -*- coding: utf-8 -*-

import json
import logging
import os
from os.path import join as pjoin
import shutil

import torch

from dwi_ml.experiment.learning_utils import compute_gradient_norm
from dwi_ml.experiment.memory import log_gpu_memory_usage
from dwi_ml.model.experiments import ExperimentAbstract
from dwi_ml.model.direction_getter_models import (
    KEYS_TO_DIRECTION_GETTER_MODEL, AbstractDirectionGetterModel)

from Learn2Track.model.embeddings import EmbeddingAbstract
from Learn2Track.model.stacked_rnn import StackedRNN
from Learn2Track.model.learn2track_model import Learn2TrackModel

DEFAULT_LAYERS = [100]
LEARNING_RATE = 1e-3
VERSION = 0


class Learn2TrackExperiment(ExperimentAbstract):
    def __init__(self,
                 previous_dir_embedding_model: EmbeddingAbstract,
                 rnn_model: Learn2TrackModel,
                 direction_getter_model: AbstractDirectionGetterModel,
                 layers, dropout,
                 use_skip_connections, use_layer_normalization, clip_grad,
                 weight_decay, input_size, nb_previous_dirs, saving_dir):
        """
        rnn_model : str
            RNN model to use;
            Choices available are : [lstm | gru]
        direction_getter_model : str
            Model converting RNN outputs into directions. Choices available
            are : [regression | sphere-classification | gaussian-mixture]
        layers : list of int
            List of hidden layers sizes. If None, the default [100] will be
            applied.
        dropout : float
            Dropout probability to use in LSTM hidden layers.
        use_skip_connections : bool
            Add skip connections from the input to all hidden layers, and from
            all hidden layers to the output layer.
        use_layer_normalization : bool
            Apply layer normalization after each RNN layer.
        clip_grad : float
            Clip the gradient norm of the model's parameters to address the
            exploding gradient problem in RNNs
        weight_decay : float
            Add a weight decay penalty on the parameters
        input_size:
            Size of the input to the model. This means that you must wait until
            the data has been loaded before instantiating this model.
        saving_dir: str
            Path where to save the model's final state when using self.save().
            Final path will be saving_dir/model
        """
        super().__init__()

        # Init properties
        self.rnn_model = rnn_model.lower()
        self.direction_getter_model = direction_getter_model.lower()
        self.layers = layers if layers else DEFAULT_LAYERS
        self.dropout = dropout
        self.clip_grad = clip_grad
        self.weight_decay = weight_decay
        self.use_skip_connections = use_skip_connections
        self.use_layer_normalization = use_layer_normalization
        self.input_size = input_size
        self.nb_previous_dirs = nb_previous_dirs
        self.saving_dir = saving_dir

        # Will be build when building model. Depends on paramaters
        self.optimizer = None
        self.model = None
        self.best_model_state = None
        self.current_grad_norm = None

        self._build_model()

    def _build_model(self):
        """
        Build PyTorch LSTM model
        """
        # `input_size` is inferred from the dataset

        direction_getter_cls = KEYS_TO_DIRECTION_GETTER_MODEL[
            self.direction_getter_model]

        # Assert that the output model supports compressed streamlines
        if (self.step_size is None and
                not direction_getter_cls.supportsCompressedStreamlines):
            raise ValueError(
                "Direction getter model '{}' does not support compressed "
                "streamlines. Please specifiy a step size.".format(
                    self.direction_getter_model))

        stacked_rnn = StackedRNN(
            rnn_base=self.rnn_model, input_size=self.input_size,
            layer_sizes=self.layers,
            use_skip_connections=self.skip_connections,
            use_layer_normalization=self.use_layer_normalization,
            dropout=self.dropout)
        direction_getter_model = direction_getter_cls(
            hidden_size=stacked_rnn.output_size,
            dropout=self.dropout)

        # Main Model = RNN + Direction getter
        self.model = Learn2TrackExperiment(stacked_rnn,
                                           direction_getter_model,
                                           self.add_previous_dir)

        # Send model to device
        # NOTE: This ordering is important! The optimizer needs to use the cuda
        # Tensors if using the GPU...
        self.model.to(device=self.device)

        # Build optimizer (Optimizer is built here since it needs the model
        # parameters)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=LEARNING_RATE,
                                          weight_decay=self.weight_decay)

        # Initialize best model
        # Uses torch's module state_dict.
        self.best_model_state = self.model.state_dict()

    @property
    def hyperparameters(self):
        hyperparameters = {
            'rnn_model': self.rnn_model,
            'direction_getter_model': self.direction_getter_model,
            'layers': self.layers,
            'dropout': self.dropout,
            'skip_connections': self.use_skip_connections,
            'layer_normalization': self.use_layer_normalization
        }
        return hyperparameters

    @property
    def attributes(self):
        """All parameters necessary to create again the same model"""
        params = {
            'model_version': VERSION,
            'saving_dir': self.saving_dir,
            'best_model_state': self.best_model_state,
        }
        return params

    def run_model_and_compute_loss(self, input_data,
                                   cpu_computations_were_avoided: bool = True,
                                   is_training: bool = False) -> float:
        """Run a batch of data through the model and return the mean loss.

        Parameters
        ----------
        input_data : tuple of (List, dict)
            This is the output of the BatchSequencesSampleOneInputVolume's
            load_batch() function. If avoid_cpu_computations, data is
            (batch_streamlines, final_streamline_ids_per_subj).
            Else, data is (packed_inputs, packed_directions).
        cpu_computations_were_avoided: bool
            Batch sampler's avoid_cpu_computation option value at the moment of
            loading the batch data.
        is_training : bool
            If True, record the computation graph and backprop through the
            model parameters.

        Returns
        -------
        mean_loss : float
            The mean loss of the provided batch
        total_norm: float
            The total norm (sqrt(sum(params**2)) of parameters before gradient
            clipping, if any.
        """
        if is_training:
            # If training, enable gradients for backprop.
            # Uses torch's module train(), which sets the module in training
            # mode.
            self.model.train()
            grad_context = torch.enable_grad
        else:
            # If evaluating, turn gradients off.
            # Uses torch's module train(), which sets the module in evalutation
            # mode.
            self.model.eval()
            grad_context = torch.no_grad

        with grad_context():
            if cpu_computations_were_avoided:
                # Data interpolation has not been done yet. GPU computations
                # need to be done here in the main thread. Running final steps
                # of data preparation.
                logging.debug('Finalizing input data preparation on GPU.')
                batch_streamlines, streamline_ids_per_subj = input_data
                packed_directions = \
                    self.train_batch_sampler.compute_and_normalize_directions(
                        batch_streamlines, streamline_ids_per_subj)
                packed_inputs = \
                    self.train_batch_sampler.compute_interpolation(
                        batch_streamlines, streamline_ids_per_subj,
                        packed_directions)
            else:
                # Data is already ready and packed
                packed_inputs, packed_targets = input_data

            if self.use_gpu:
                packed_inputs = packed_inputs.cuda()
                packed_targets = packed_targets.cuda()

            logging.debug("Running model on a batch of {} streamlines"
                          .format(packed_inputs.batch_sizes[0]))

            if is_training:
                # Reset parameter gradients
                self.model.optimizer.zero_grad()

            try:
                # Apply model. This calls our RecurrentTrackingModel's
                # forward function, which returns the outputs and new_states.
                model_outputs, _ = self.model(packed_inputs)
            except RuntimeError:
                # Training RNNs with variable-length sequences on the GPU can
                # cause memory fragmentation in the pytorch-managed cache,
                # possibly leading to "random" OOM RuntimeError during
                # training. Emptying the GPU cache seems to fix the problem for
                # now. We don't do it every update because it can be time
                # consuming.
                torch.cuda.empty_cache()
                model_outputs, _ = self.model(packed_inputs)

            # Compute loss
            mean_loss = self.model.compute_loss(model_outputs,
                                                packed_targets)

            if is_training:
                # Backprop loss
                logging.debug('Computing back propagation')
                mean_loss.backward()

                # Clip gradient if necessary before updating parameters
                # Remembering unclipped value.
                if self.clip_grad:
                    self.current_grad_norm = compute_gradient_norm(
                        self.model.parameters())
                    logging.debug("Gradient norm: {}"
                                  .format(self.current_grad_norm))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip_grad)

                # Update parameters
                self.optimizer.step()

            if self.use_gpu:
                log_gpu_memory_usage()

        return mean_loss.cpu().item()

    def save(self):
        """
        This method should be called at the end of training to save the best
        parameters for the model.
        """
        # Make model directory
        model_dir = pjoin(self.saving_dir, "model")

        # If a model was already saved, back it up and erase it after saving
        # the new.
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = pjoin(self.saving_dir, "model_old")
            shutil.move(model_dir, to_remove)
        os.makedirs(model_dir)

        # Save attributes
        with open(pjoin(model_dir, "attributes.json"), 'w') as json_file:
            json_file.write(
                json.dumps(self.attributes, indent=4, separators=(',', ': ')))

        # Save hyperparams
        with open(pjoin(model_dir, "hyperparameters.json"), 'w') as json_file:
            json_file.write(json.dumps(self.hyperparameters, indent=4,
                                       separators=(',', ': ')))

        # Save model
        torch.save(self.best_model_state,
                   pjoin(model_dir, "best_model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)
