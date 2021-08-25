# -*- coding: utf-8 -*-
import logging
from typing import Any, Tuple, Union, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from dwi_ml.model.direction_getter_models import AbstractDirectionGetterModel
from dwi_ml.model.main_models import ModelAbstract

from Learn2Track.model.stacked_rnn import StackedRNN
from Learn2Track.model.embeddings import EmbeddingAbstract
from Learn2Track.utils.packed_sequences import (unpack_sequence,
                                                unpack_tensor_from_indices)


class Learn2TrackModel(ModelAbstract):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, a RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self,
                 previous_dir_embedding_model: EmbeddingAbstract,
                 input_embedding_model: EmbeddingAbstract,
                 rnn_model: StackedRNN,
                 direction_getter_model: AbstractDirectionGetterModel):
        """
        Parameters
        ----------
        previous_dir_embedding_model: EmbeddingAbstract
            Instantiated model for the previous directions embedding. Outputs
            will be concatenated to inputs before entering the rnn_model.
            See examples in Learn2track.model.embeddings.
        input_embedding_model: EmbeddingAbstract
            Instantiated model for the input embedding.
            See examples in Learn2track.model.embeddings.
        rnn_model : StackedRNN
            Instantiated recurrent model to process sequences. The StackedRNN
            is composend of RNN + normalization + dropout + relu + skip
            connections.
        direction_getter_model : AbstractDirectionGetterModel
            Instantiated model used to convert the RNN outputs into a
            direction. See dwi_ml.model.direction_getter_models for model
            descriptions.
        """
        super().__init__()

        self.prev_dir_embedding = previous_dir_embedding_model
        self.input_embedding = input_embedding_model
        self.rnn_model = rnn_model
        self.direction_getter = direction_getter_model

    @property
    def hyperparameters(self):
        return {}

    @property
    def attributes(self):
        attributes = {
            'prev_dir_embedding': self.prev_dir_embedding.attributes,
            'input_embedding': self.input_embedding.attributes,
            'rnn_model': self.rnn_model.attributes,
            'direction_getter': self.direction_getter_model.attributes,
        }
        return attributes

    def forward(self, inputs: Union[PackedSequence, Tensor],
                prev_dirs: Union[PackedSequence, List[Tensor], Tensor],
                hidden_states: Any = None) -> Tuple[Any, Any]:
        """Run the model on a batch of sequences.

        Parameters
        ----------
        inputs : torch.Tensor or PackedSequence
            Batch of input sequences.
        prev_dirs: torch.Tensor or PackedSequence or List[Tensors]
        hidden_states : Any
            The current hidden states of the (stacked) RNN model.

        Returns
        -------
        outputs : Any
            Output data, ready to be passed to either `compute_loss()` or
            `get_tracking_directions()`.
        out_hidden_states : Any
            The last step hidden states (h_(t-1), C_(t-1) for LSTM) for each
            layer.
        """
        orig_input = inputs

        # Previous dirs embedding, input_dirs embedding
        logging.debug("================ 1. Previous dir embedding...")
        prev_dirs = self.prev_dir_embedding(prev_dirs)
        logging.debug("================ 2. Inputs embedding...")
        inputs = self.input_embedding(inputs)

        # Concatenating this result to input and packing if list
        logging.debug("================ 3. Concatenating previous dirs and "
                      "inputs's embeddings")
        inputs = self._concat_prev_dirs(inputs, prev_dirs)

        # Run the inputs sequences through the stacked RNN
        logging.debug("================ 4. Stacked RNN...")
        rnn_output, out_hidden_states = self.rnn_model(inputs, hidden_states)

        # Run the rnn outputs into the direction getter model
        logging.debug("================ 5. Direction getter.")
        if (isinstance(orig_input, PackedSequence) or
                isinstance(orig_input, list)):
            # rnn_output is a packed sequence. Sending only the tensor version
            # then recreating the packed sequence
            data = self.direction_getter(rnn_output.data)
            logging.debug("Direction getter output: {}".format(data))
            if isinstance(orig_input, PackedSequence):
                final_directions = PackedSequence(data, inputs.batch_sizes,
                                                  inputs.sorted_indices,
                                                  inputs.unsorted_indices)
            else:  # isinstance(orig_input, list):
                # Reordering to fit original list
                indices = unpack_sequence(inputs, get_indices_only=True)
                final_directions = unpack_tensor_from_indices(data, indices)
        else:
            final_directions = self.direction_getter(rnn_output)
        logging.debug("Final directions: {}".format(final_directions))

        # 4. Return the hidden states. Necessary for the generative
        # (tracking) part, done step by step.
        return final_directions, out_hidden_states

    @staticmethod
    def _concat_prev_dirs(inputs, prev_dirs):
        """Concatenating data depends on the data type."""
        if isinstance(inputs, Tensor) and isinstance(prev_dirs, Tensor):
            inputs = torch.cat((inputs, prev_dirs), dim=-1)
            logging.debug("Previous dir and inputs both seem to be tensors. "
                          "Concatenating if dimensions fit. Input shape: {},"
                          "Prev dir shape: {}"
                          .format(inputs.shape, prev_dirs.shape))
            inputs = torch.cat((inputs, prev_dirs), dim=-1)
            logging.debug("Concatenated shape: {}".format(inputs.shape))

        elif isinstance(inputs, list) and isinstance(prev_dirs, list):
            nb_s_input = len(inputs)
            nb_s_prev_dir = len(prev_dirs)
            logging.debug("Previous dir and inputs both seem to be a list of "
                          "tensors, probably one per streamline. Now checking "
                          "that their dimensions fit. Nb inputs: {}, Nb "
                          "prev_dirs: {}".format(nb_s_input, nb_s_prev_dir))
            assert nb_s_input == nb_s_prev_dir, \
                "Both lists do not have the same length (not the same " \
                "number of streamlines?)"
            inputs = [torch.cat((inputs[i], prev_dirs[i]), dim=-1) for
                      i in range(nb_s_input)]

            logging.debug("Packing inputs.")
            inputs = pack_sequence(inputs, enforce_sorted=False)

        elif (isinstance(inputs, PackedSequence) and
                isinstance(prev_dirs, PackedSequence)):
            logging.debug("Previous dirs and inputs are both PackedSequence. "
                          "Trying to concatenate data. Input shape: {},"
                          "prev_dir shape: {}"
                          .format(inputs.data.shape, prev_dirs.data.shape))
            if (inputs.unsorted_indices is None or
                    prev_dirs.unsorted_indices is None):
                raise ValueError("Packed sequences 'unsorted_indices' param "
                                 "is None. You have probably created your "
                                 "PackedSequence using enforce_sorted=True. "
                                 "Please use False.")
            nb_s_input = len(inputs.unsorted_indices)
            nb_s_prev_dir = len(prev_dirs.unsorted_indices)
            assert nb_s_input == nb_s_prev_dir, \
                "Both lists do not have the same length (not the same " \
                "number of streamlines?)"
            new_input_data = torch.cat((inputs.data, prev_dirs.data), dim=1)

            logging.debug("Packing concatenated inputs.")
            inputs = PackedSequence(new_input_data, inputs.batch_sizes,
                                    inputs.sorted_indices,
                                    inputs.unsorted_indices)
        else:
            raise ValueError("Could not concatenate previous_dirs and inputs."
                             "Currently, we expect previous_dir to fit inputs"
                             "type (Tensor or list of Tensors or "
                             "PackedSequence)")

        return inputs

    def compute_loss(self, outputs: Any,
                     targets: Union[PackedSequence, Tensor]) -> Tensor:
        """
        Computes the loss function using the provided outputs and targets.
        Returns the mean loss (loss averaged across timesteps and sequences).

        Parameters
        ----------
        outputs : Any
            The model outputs for a batch of sequences. Ex: a gaussian mixture
            direction getter returns a Tuple[Tensor, Tensor, Tensor], but a
            cosine regression direction getter return a simple Tensor. Please
            make sure that the chosen direction_getter's output size fits with
            the target ou the target's data if it's a PackedSequence.
        targets : PackedSequence or torch.Tensor
            The target values for the batch

        Returns
        -------
        mean_loss : torch.Tensor
            The loss between the outputs and the targets, averaged across
            timesteps and sequences.
        """
        if isinstance(targets, PackedSequence):
            targets = targets.data
        mean_loss = self.direction_getter.compute_loss(outputs, targets)
        return mean_loss

    def sample_tracking_directions(self, single_step_inputs: Tensor,
                                   states: Any) -> Tuple[Tensor, Any]:
        """
        Runs a batch of single-step inputs through the model, then get the
        tracking directions. E.g. for probabilistic models, we need to sample
        the tracking directions.

        Parameters
        ----------
        single_step_inputs : torch.Tensor
            A batch of single-step inputs. Should be of size
            [batch_size x 1 x n_features] where batch_size is ????
            the number of streamline or the number of timesteps???
                                                                                                                                # toDo
        states : Any
            The current hidden states for the RNN model.

        Returns
        -------
        directions : torch.Tensor
            The predicted/sampled directions
        rnn_recurrent_output : Any
            The hidden states for the next tracking step
        """

        # Call directly the forward function
        outputs, rnn_recurrent_output = \
            self.__call__(single_step_inputs[:, None, :], states)

        directions = self.direction_getter.sample_directions(outputs)
        directions = directions.reshape((single_step_inputs.shape[0], 3))
        return directions, rnn_recurrent_output
