# -*- coding: utf-8 -*-
import logging
from typing import Any, Tuple, Union, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.main_models import MainModelAbstract

from Learn2Track.models.stacked_rnn import StackedRNN


class Learn2TrackModel(MainModelAbstract):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, a RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self, nb_previous_dirs: int, prev_dirs_embedding_size: int,
                 prev_dirs_embedding_key, input_size: int, input_embedding_key,
                 input_embedding_size_ratio: float, rnn_key: str,
                 rnn_layer_sizes: List[int], use_skip_connection: bool,
                 use_layer_normalization: bool, dropout: float,
                 direction_getter_key, **_):
        """
        Params
        ------
        input_size: int
            This value should be known from the actual data.
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received.
        prev_dirs_embedding_size: int
            How to transform prev_dir inputs (dimension = 3 * nb_previous_dirs)
            during embedding. Total embedding size will be nb_previous_dirs *
            user-given prev_dirs_embedding_size.
        prev_dirs_embedding_cls: a EmbeddingAbstract class
            One of the classes in Learn2track.model.embeddings
        input_embedding_cls: idem
        input_embedding_size_ratio: float
            Considering as the input size is not known beforehand, not asking
            user for the embedding size but rather for a ratio. Ex: 0.5 means
            the embedding size will be half the input size.
        rnn_key: either 'LSTM' or 'GRU'
        rnn_layer_sizes: List[int]
            The list of layer sizes for the rnn. The real size will also depend
            on the skip_connection and layer_normalization parameters.
        use_skip_connection: bool

        use_layer_normalization: bool

        """
        super().__init__()

        # 0. Verifying keys
        print("prev_dirs_embedding_key: {}".format(prev_dirs_embedding_key))
        if prev_dirs_embedding_key:
            prev_dirs_emb_cls = keys_to_embeddings[prev_dirs_embedding_key]
        else:
            prev_dirs_emb_cls = None
        input_embedding_cls = keys_to_embeddings[input_embedding_key]
        direction_getter_cls = keys_to_direction_getters[direction_getter_key]
        # and rnn_key will be checked in stacked_rnn.

        # 1. Previous dir embedding
        self.nb_previous_dirs = nb_previous_dirs
        if nb_previous_dirs > 0:
            self.prev_dirs_embedding_size = prev_dirs_embedding_size
            self.prev_dirs_embedding = prev_dirs_emb_cls(
                input_size=nb_previous_dirs * 3,
                output_size=self.prev_dirs_embedding_size)
        else:
            self.prev_dirs_embedding_size = 0
            self.prev_dirs_embedding = None

        # 2. Input embedding
        self.input_size = input_size
        self.input_embedding_size_ratio = input_embedding_size_ratio
        input_embedding_size = int(input_size * input_embedding_size_ratio)
        self.input_embedding = input_embedding_cls(
            input_size=input_size, output_size=input_embedding_size)

        # 3. Stacked RNN
        rnn_input_size = self.prev_dirs_embedding_size + input_embedding_size
        self.use_skip_connection = use_skip_connection
        self.use_layer_normalization = use_layer_normalization
        self.rnn_key = rnn_key
        self.rnn_layer_sizes = rnn_layer_sizes
        self.dropout = dropout
        self.rnn_model = StackedRNN(
            rnn_key, rnn_input_size, rnn_layer_sizes,
            use_skip_connections=use_skip_connection,
            use_layer_normalization=use_layer_normalization, dropout=dropout)

        # 4. Direction getter
        # toDo: add parameters. Ex: dropout and nb_gaussians
        self.direction_getter = direction_getter_cls(
            self.rnn_model.output_size)

    @property
    def params(self):
        # Every parameter necessary to build the different layers again.
        # converting np.int64 to int to allow json dumps.
        params = {
            'prev_dirs_embedding':
                self.prev_dirs_embedding.params if
                self.prev_dirs_embedding else None,
            # This should contain the input size:
            'input_embedding_size_ratio': self.input_embedding_size_ratio,
            'input_embedding': self.input_embedding.params,
            'rnn_model': self.rnn_model.params,
            'direction_getter': self.direction_getter.params,
        }
        return params

    def set_log(self, log: logging.Logger):
        """Possibility to pass a tqdm-compatible logger in case the dataloader
        is iterated through a tqdm progress bar. Note that, of course, log
        outputs will be confusing, particularly in debug mode, considering
        that the dataloader may use more than one method in parallel."""
        self.log = log
        self.input_embedding.set_log(log)
        if self.prev_dirs_embedding:
            self.prev_dirs_embedding.set_log(log)
        self.rnn_model.set_log(log)
        self.direction_getter.set_log(log)

    def forward(self, inputs: PackedSequence, prev_dirs: PackedSequence,
                hidden_states: Any = None) -> Tuple[Any, Any]:
        """Run the model on a batch of sequences.

        Parameters
        ----------
        inputs: PackedSequence
            Batch of input sequences, i.e. MRI data. We expect both inputs and
            prev_dirs to be packedSequence based on current trainer
            implementation.
        prev_dirs: PackedSequence
            Batch of past directions
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
        orig_inputs = inputs

        # Previous dirs embedding, input_dirs embedding
        self.log.debug("================ 1. Previous dir embedding, if any "
                       "(on tensor)...")
        if prev_dirs is not None:
            self.log.debug("Input size: {}".format(prev_dirs.data.shape[-1]))
            prev_dirs = self.prev_dirs_embedding(prev_dirs.data)
            self.log.debug("Output size: {}".format(prev_dirs.shape[-1]))

        self.log.debug("================ 2. Inputs embedding (on tensor)...")
        self.log.debug("Input size: {}".format(inputs.data.shape[-1]))
        inputs = self.input_embedding(inputs.data)
        self.log.debug("Output size: {}".format(inputs.shape[-1]))

        # Concatenating this result to input and packing if list
        self.log.debug("================ 3. Concatenating previous dirs and "
                       "inputs's embeddings")
        if prev_dirs is not None:
            inputs = self._concat_prev_dirs(inputs, prev_dirs)

        self.log.debug("Packing back data for RNN.")
        inputs = PackedSequence(inputs, orig_inputs.batch_sizes,
                                orig_inputs.sorted_indices,
                                orig_inputs.unsorted_indices)

        # Run the inputs sequences through the stacked RNN
        self.log.debug("================ 4. Stacked RNN....")
        rnn_output, out_hidden_states = self.rnn_model(inputs, hidden_states)
        self.log.debug("Output size: {}".format(rnn_output.data.shape[-1]))

        # Run the rnn outputs into the direction getter model
        self.log.debug("================ 5. Direction getter.")
        final_directions = self.direction_getter(rnn_output.data)
        self.log.debug("Output size: {}".format(final_directions.shape[-1]))

        # 4. Return the hidden states. Necessary for the generative
        # (tracking) part, done step by step.
        return final_directions, out_hidden_states

    def _concat_prev_dirs(self, inputs, prev_dirs):
        """Concatenating data depends on the data type."""

        if isinstance(inputs, Tensor) and isinstance(prev_dirs, Tensor):
            self.log.debug("Previous dir and inputs both seem to be tensors. "
                           "Concatenating if dimensions fit. Input shape: {},"
                           "Prev dir shape: {}"
                           .format(inputs.shape, prev_dirs.shape))
            inputs = torch.cat((inputs, prev_dirs), dim=-1)
            self.log.debug("Concatenated shape: {}".format(inputs.shape))

        elif isinstance(inputs, list) and isinstance(prev_dirs, list):
            nb_s_input = len(inputs)
            nb_s_prev_dir = len(prev_dirs)
            self.log.debug("Previous dir and inputs both seem to be a list of "
                           "tensors, probably one per streamline. Now "
                           "checking that their dimensions fit. Nb inputs: "
                           "{}, Nb prev_dirs: {}"
                           .format(nb_s_input, nb_s_prev_dir))
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
        tracking directions. E.g. for probabilistic model, we need to sample
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
