# -*- coding: utf-8 -*-
import logging
from typing import Any, Union, List, Iterable

import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.main_models import MainModelWithPD

from Learn2Track.models.stacked_rnn import StackedRNN


class Learn2TrackModel(MainModelWithPD):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, a RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self, experiment_name,
                 # PREVIOUS DIRS
                 nb_previous_dirs: int, prev_dirs_embedding_size: int,
                 prev_dirs_embedding_key: str,
                 # INPUTS
                 nb_features: int, input_embedding_key: str,
                 input_embedding_size: int, input_embedding_size_ratio: float,
                 # RNN
                 rnn_key: str, rnn_layer_sizes: List[int],
                 use_skip_connection: bool, use_layer_normalization: bool,
                 dropout: float,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: dict,
                 # Other
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, Iterable[float], None],
                 normalize_directions: bool):
        """
        Params
        ------
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received.
        prev_dirs_embedding_size: int
            How to transform prev_dir inputs (dimension = 3 * nb_previous_dirs)
            during embedding. Total embedding size will be nb_previous_dirs *
            user-given prev_dirs_embedding_size. If None, embedding_size will
            be set to 3*nb_previous_dirs.
        prev_dirs_embedding_key: str,
            Key to a embedding class (one of
            dwi_ml.models.embeddings_on_tensors.keys_to_embeddings)
        input_size: int
            This value should be known from the actual data. Number of features
            in the data (last dimension).
        input_embedding_key: str
            Key to a embedding class (one of
            dwi_ml.models.embeddings_on_tensors.keys_to_embeddings)
        input_embedding_size: int
            Output embedding size for the input. If None, will be set to
            input_size.
        input_embedding_size_ratio: float
            Other possibility to define input_embedding_size, which then equals
            [ratio * (nb_features * (nb_neighbors+1))]
        rnn_key: either 'LSTM' or 'GRU'
        rnn_layer_sizes: List[int]
            The list of layer sizes for the rnn. The real size will depend
            on the skip_connection parameter.
        use_skip_connection: bool
            Whether to use skip connections. See [1] (Figure 1) to visualize
            the architecture.
        use_layer_normalization: bool
            Wheter to apply layer normalization to the forward connections. See
            [2].
        dropout : float
            If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with given dropout probability.
        dg_key: str
            Key to a direction getter class (one of
            dwi_ml.direction_getter_models.keys_to_direction_getters).
        dg_args: dict
            Arguments necessary for the instantiation of the chosen direction
            getter (other than input size, which will be the rnn's output
            size).
        neighborhood_type: str
            The type of neighborhood to add. One of 'axes', 'grid' or None. If
            None, don't add any. See
            dwi_ml.data.processing.space.Neighborhood for more information.
        neighborhood_radius : Union[int, float, Iterable[float]]
            Add neighborhood points at the given distance (in voxels) in each
            direction (nb_neighborhood_axes). (Can be none)
                - For a grid neighborhood: type must be int.
                - For an axes neighborhood: type must be float. If it is an
                iterable of floats, we will use a multi-radius neighborhood.
        normalize_directions: bool
            If true, direction vectors are normalized (norm=1). If the step
            size is fixed, it shouldn't make any difference. If streamlines are
            compressed, in theory you should normalize, but you could hope that
            not normalizing could give back to the algorithm a sense of
            distance between points.
        ---
        [1] https://arxiv.org/pdf/1308.0850v5.pdf
        [2] https://arxiv.org/pdf/1607.06450.pdf
        """
        super().__init__(experiment_name, nb_previous_dirs,
                         normalize_directions, neighborhood_type,
                         neighborhood_radius)

        self.prev_dirs_embedding_key = prev_dirs_embedding_key
        self.prev_dirs_embedding_size = prev_dirs_embedding_size
        self.input_embedding_key = input_embedding_key
        self.input_embedding_size = input_embedding_size
        self.input_embedding_size_ratio = input_embedding_size_ratio
        self.nb_features = nb_features
        self.use_skip_connection = use_skip_connection
        self.use_layer_normalization = use_layer_normalization
        self.rnn_key = rnn_key
        self.rnn_layer_sizes = rnn_layer_sizes
        self.dropout = dropout
        self.dg_key = dg_key
        self.dg_args = dg_args

        # 1. Previous dir embedding
        if self.nb_previous_dirs > 0:
            if prev_dirs_embedding_size is None:
                self.prev_dirs_embedding_size = nb_previous_dirs * 3
            prev_dirs_emb_cls = keys_to_embeddings[prev_dirs_embedding_key]
            self.prev_dirs_embedding = prev_dirs_emb_cls(
                input_size=nb_previous_dirs * 3,
                output_size=self.prev_dirs_embedding_size)
        else:
            if self.prev_dirs_embedding_size:
                logging.warning("Previous dirs embedding size was defined but "
                                "not previous directions are used!")
            self.prev_dirs_embedding = None

        # 2. Input embedding
        nb_neighbors = len(self.neighborhood_points) if \
            self.neighborhood_points else 0
        self.input_size = nb_features * (nb_neighbors + 1)
        if not (input_embedding_size or input_embedding_size_ratio):
            input_embedding_size = self.input_size
        if input_embedding_size and input_embedding_size_ratio:
            raise ValueError(
                "You must only give one value, either input_embedding_size or "
                "input_embedding_size_ratio")
        if input_embedding_size:
            input_embedding_size = input_embedding_size
        if input_embedding_size_ratio:
            input_embedding_size = int(self.input_size *
                                       input_embedding_size_ratio)
        input_embedding_cls = keys_to_embeddings[input_embedding_key]
        self.input_embedding = input_embedding_cls(
            input_size=self.input_size,
            output_size=input_embedding_size)

        # 3. Stacked RNN
        rnn_input_size = input_embedding_size
        if self.nb_previous_dirs > 0:
            rnn_input_size += self.prev_dirs_embedding_size
        self.rnn_model = StackedRNN(
            rnn_key, rnn_input_size, rnn_layer_sizes,
            use_skip_connections=use_skip_connection,
            use_layer_normalization=use_layer_normalization, dropout=dropout,
            logger=self.logger)

        # 4. Direction getter
        direction_getter_cls = keys_to_direction_getters[dg_key]
        self.direction_getter = direction_getter_cls(
            self.rnn_model.output_size, **dg_args)

    @property
    def params_per_layer(self):
        params = {
            'prev_dirs_embedding':
                self.prev_dirs_embedding.params if
                self.prev_dirs_embedding else None,
            'input_embedding': self.input_embedding.params,
            'rnn_model': self.rnn_model.params,
            'direction_getter': self.direction_getter.params
        }
        return params

    @property
    def params(self):
        # Every parameter necessary to build the different layers again.
        # during checkpoint state saving.

        params = super().params

        params.update({
            'prev_dirs_embedding_size': self.prev_dirs_embedding_size,
            'prev_dirs_embedding_key': self.prev_dirs_embedding_key,
            'nb_features': int(self.nb_features),
            'input_embedding_key': self.input_embedding_key,
            'input_embedding_size': int(self.input_embedding_size) if
            self.input_embedding_size else None,
            'input_embedding_size_ratio': self.input_embedding_size_ratio,
            'rnn_key': self.rnn_key,
            'rnn_layer_sizes': self.rnn_layer_sizes,
            'use_skip_connection': self.use_skip_connection,
            'use_layer_normalization': self.use_layer_normalization,
            'dropout': self.dropout,
            'dg_key': self.dg_key,
            'dg_args': self.dg_args,
        })

        return params

    def forward(self, inputs: List[torch.tensor],
                n_prev_dirs: List[torch.tensor], device,
                hidden_reccurent_states: tuple = None):
        """Run the model on a batch of sequences.

        Parameters
        ----------
        inputs: List[torch.tensor]
            Batch of input sequences, i.e. MRI data. Length of the list is the
            number of streamlines in the batch. Each tensor is of size
            [nb_points, nb_features].
        n_prev_dirs: List[torch.tensor],
            Batch of n past directions. Length of the list is the number of
            streamlines in the batch. Each tensor is of size
            [nb_points, nb_previous_dirs * 3].
        device: torch device
        hidden_reccurent_states : tuple
            The current hidden states of the (stacked) RNN model.

        Returns
        -------
        model_outputs : Any
            Output data, ready to be passed to either `compute_loss()` or
            `get_tracking_directions()`.
        out_hidden_recurrent_states : tuple
            The last steps hidden states (h_n, C_n for LSTM) for each layer.
            Tuple containing nb_layer tuples of 2 tensors (h_n, c_n) with
            shape(h_n) = shape(c_n) = [1, nb_streamlines, layer_output_size]
        """

        # Packing everything and saving info
        inputs = pack_sequence(inputs, enforce_sorted=False).to(device)

        if n_prev_dirs is not None:  # self.nb_previous_dirs should be > 0
            n_prev_dirs = \
                pack_sequence(n_prev_dirs, enforce_sorted=False).to(device)

        batch_sizes = inputs.batch_sizes
        sorted_indices = inputs.sorted_indices
        unsorted_indices = inputs.unsorted_indices

        # RUNNING THE MODEL
        self.logger.debug("================ 1. Previous dir embedding, if any "
                          "(on packed_sequence's tensor!)...")
        if n_prev_dirs is not None:  # self.nb_previous_dirs should be > 0
            self.logger.debug(
                "Input size: {}".format(n_prev_dirs.data.shape[-1]))
            n_prev_dirs = self.prev_dirs_embedding(n_prev_dirs.data)
            self.logger.debug("Output size: {}".format(n_prev_dirs.shape[-1]))

        self.logger.debug(
            "================ 2. Inputs embedding (on packed_sequence's "
            "tensor!)...")
        self.logger.debug("Input size: {}".format(inputs.data.shape[-1]))
        inputs = self.input_embedding(inputs.data)
        self.logger.debug("Output size: {}".format(inputs.shape[-1]))

        self.logger.debug(
            "================ 3. Concatenating previous dirs and inputs's "
            "embeddings")
        if n_prev_dirs is not None:
            inputs = self._concat_prev_dirs(inputs, n_prev_dirs)
        inputs = PackedSequence(inputs, batch_sizes, sorted_indices,
                                unsorted_indices)

        self.logger.debug("================ 4. Stacked RNN....")
        rnn_output, out_hidden_recurrent_states = self.rnn_model(
            inputs, hidden_reccurent_states)
        self.logger.debug("Output size: {}".format(rnn_output.data.shape[-1]))

        self.logger.debug("================ 5. Direction getter.")
        model_outputs = self.direction_getter(rnn_output.data)
        self.logger.debug("Output size: {}".format(model_outputs.shape[-1]))

        # Return the hidden states. Necessary for the generative
        # (tracking) part, done step by step.
        return model_outputs, out_hidden_recurrent_states

    def _concat_prev_dirs(self, inputs, prev_dirs):
        """Concatenating data depends on the data type."""

        assert isinstance(inputs, torch.Tensor), "Inputs should be a tensor"
        assert isinstance(prev_dirs, torch.Tensor), \
            "prev_dirs should be a tensor"

        self.logger.debug(
            "Previous dir and inputs both seem to be tensors. "
            "Concatenating if dimensions fit. Input shape: {},"
            "Prev dir shape: {}"
            .format(inputs.shape, prev_dirs.shape))
        inputs = torch.cat((inputs, prev_dirs), dim=-1)
        self.logger.debug("Concatenated shape: {}".format(inputs.shape))

        return inputs

    def compute_loss(self, model_outputs: Any, targets: list, device):
        """
        Computes the loss function using the provided outputs and targets.
        Returns the mean loss (loss averaged across timesteps and sequences).

        Parameters
        ----------
        model_outputs : Any
            The model outputs for a batch of sequences. Ex: a gaussian mixture
            direction getter returns a Tuple[Tensor, Tensor, Tensor], but a
            cosine regression direction getter return a simple Tensor. Please
            make sure that the chosen direction_getter's output size fits with
            the target ou the target's data if it's a PackedSequence.
        targets : List
            The target values for the batch (the directions). In super, the
            streamlines are sent and we compute the directions here. But since
            we have already done that to prepare the previous dirs, sending
            them here.
        device: torch device

        Returns
        -------
        mean_loss : torch.Tensor
            The loss between the outputs and the targets, averaged across
            timesteps and sequences.
        """
        # Packing dirs and using the .data
        targets = pack_sequence(targets, enforce_sorted=False).data.to(device)

        # Computing loss
        mean_loss = self.direction_getter.compute_loss(model_outputs, targets)

        return mean_loss

    def get_tracking_direction_det(self, model_outputs):
        next_dir = self.direction_getter.get_tracking_direction_det(
            model_outputs)
        next_dir = next_dir.detach().numpy().squeeze()
        return next_dir

    def sample_tracking_direction_prob(self, model_outputs):
        logging.debug("Getting a deterministic direction from {}"
                      .format(type(self.direction_getter)))
        return self.direction_getter.sample_tracking_direction_prob(
            model_outputs)
