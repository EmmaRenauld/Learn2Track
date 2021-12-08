# -*- coding: utf-8 -*-
import logging
from typing import Any, Tuple, Union, List, Iterable

import torch

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_and_normalize_directions, compute_n_previous_dirs
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.main_models import MainModelWithNeighborhood

from Learn2Track.models.stacked_rnn import StackedRNN


class Learn2TrackModel(MainModelWithNeighborhood):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, a RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self, experiment_name,
                 nb_previous_dirs: int, prev_dirs_embedding_size: int,
                 prev_dirs_embedding_key: str, nb_features: int,
                 input_embedding_key: str, input_embedding_size: int,
                 input_embedding_size_ratio: float,
                 rnn_key: str, rnn_layer_sizes: List[int],
                 use_skip_connection: bool, use_layer_normalization: bool,
                 dropout: float, direction_getter_key,
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
        super().__init__(experiment_name, neighborhood_type,
                         neighborhood_radius)

        self.prev_dirs_embedding_key = prev_dirs_embedding_key
        self.nb_previous_dirs = nb_previous_dirs
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
        self.normalize_directions = normalize_directions
        self.direction_getter_key = direction_getter_key

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
        rnn_input_size = self.prev_dirs_embedding_size + input_embedding_size
        self.rnn_model = StackedRNN(
            rnn_key, rnn_input_size, rnn_layer_sizes,
            use_skip_connections=use_skip_connection,
            use_layer_normalization=use_layer_normalization, dropout=dropout,
            logger=self.logger)

        # 4. Direction getter
        direction_getter_cls = keys_to_direction_getters[direction_getter_key]
        # toDo: add parameters. Ex: dropout and nb_gaussians
        self.direction_getter = direction_getter_cls(
            self.rnn_model.output_size)

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
            'nb_previous_dirs': self.nb_previous_dirs,
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
            'direction_getter_key': self.direction_getter_key,
            'normalize_directions': self.normalize_directions,
        })

        return params

    @staticmethod
    def prepare_inputs(inputs):
        """
        Inputs should be already prepared by the batch sampler (meaning the
        neighborhood is added) because it needs to interpolate data from the
        volumes. Only needs to be packed.
        """
        packed_inputs = pack_sequence(inputs, enforce_sorted=False)
        return packed_inputs

    def prepare_targets(self, streamlines, device):
        """
        Targets are the next direction at each point (packedSequence)
        """
        directions = compute_and_normalize_directions(
            streamlines, device, self.normalize_directions)
        packed_directions = pack_sequence(directions, enforce_sorted=False)
        return directions, packed_directions

    def prepare_previous_dirs(self, directions, device):
        """
        Preparing the n_previous_dirs for each point (when they don't exist,
        use zeros).
        The method returns a value for each of the n points of the streamline,
        (thus one more points than the number of directions). Here we do not
        use the last point as it does not have an associated target.
        Packing.
        """
        if self.nb_previous_dirs == 0:
            return None

        _n_previous_dirs = compute_n_previous_dirs(
            directions, self.nb_previous_dirs, device=device)

        # Not keeping the last point
        n_previous_dirs = [s[:-1] for s in _n_previous_dirs]

        # Packing.
        n_previous_dirs = pack_sequence(n_previous_dirs, enforce_sorted=False)

        return n_previous_dirs

    def forward(self, inputs: PackedSequence, prev_dirs: PackedSequence,
                hidden_reccurent_states: Any = None) -> Tuple[Any, Any]:
        """Run the model on a batch of sequences.

        Parameters
        ----------
        inputs: PackedSequence
            Batch of input sequences, i.e. MRI data. We expect both inputs and
            prev_dirs to be packedSequence based on current trainer
            implementation.
        prev_dirs: PackedSequence
            Batch of past directions
        hidden_reccurent_states : Any
            The current hidden states of the (stacked) RNN model.

        Returns
        -------
        outputs : Any
            Output data, ready to be passed to either `compute_loss()` or
            `get_tracking_directions()`.
        out_hidden_recurrent_states : Any
            The last step hidden states (h_(t-1), C_(t-1) for LSTM) for each
            layer.
        """
        orig_inputs = inputs

        # Previous dirs embedding, input_dirs embedding
        self.logger.debug("================ 1. Previous dir embedding, if any "
                          "(on packed_sequence's tensor!)...")
        if prev_dirs is not None:
            self.logger.debug(
                "Input size: {}".format(prev_dirs.data.shape[-1]))
            prev_dirs = self.prev_dirs_embedding(prev_dirs.data)
            self.logger.debug("Output size: {}".format(prev_dirs.shape[-1]))

        self.logger.debug(
            "================ 2. Inputs embedding (on "
            "packed_sequence's tensor!)...")
        self.logger.debug("Input size: {}".format(inputs.data.shape[-1]))
        inputs = self.input_embedding(inputs.data)
        self.logger.debug("Output size: {}".format(inputs.shape[-1]))

        # Concatenating this result to input and packing if list
        self.logger.debug(
            "================ 3. Concatenating previous dirs and "
            "inputs's embeddings")
        if prev_dirs is not None:
            inputs = self._concat_prev_dirs(inputs, prev_dirs)

        self.logger.debug("Packing back data for RNN.")
        inputs = PackedSequence(inputs, orig_inputs.batch_sizes,
                                orig_inputs.sorted_indices,
                                orig_inputs.unsorted_indices)

        # Run the inputs sequences through the stacked RNN
        self.logger.debug("================ 4. Stacked RNN....")
        rnn_output, out_hidden_recurrent_states = self.rnn_model(
            inputs, hidden_reccurent_states)
        self.logger.debug("Output size: {}".format(rnn_output.data.shape[-1]))

        # Run the rnn outputs into the direction getter model
        self.logger.debug("================ 5. Direction getter.")
        final_directions = self.direction_getter(rnn_output.data)
        self.logger.debug("Output size: {}".format(final_directions.shape[-1]))

        # 4. Return the hidden states. Necessary for the generative
        # (tracking) part, done step by step.
        return final_directions, out_hidden_recurrent_states

    def _concat_prev_dirs(self, inputs, prev_dirs):
        """Concatenating data depends on the data type."""

        if isinstance(inputs, Tensor) and isinstance(prev_dirs, Tensor):
            self.logger.debug(
                "Previous dir and inputs both seem to be tensors. "
                "Concatenating if dimensions fit. Input shape: {},"
                "Prev dir shape: {}"
                .format(inputs.shape, prev_dirs.shape))
            inputs = torch.cat((inputs, prev_dirs), dim=-1)
            self.logger.debug("Concatenated shape: {}".format(inputs.shape))

        elif isinstance(inputs, list) and isinstance(prev_dirs, list):
            nb_s_input = len(inputs)
            nb_s_prev_dir = len(prev_dirs)
            self.logger.debug(
                "Previous dir and inputs both seem to be a list of "
                "tensors, probably one per streamline. Now "
                "checking that their dimensions fit. Nb inputs: "
                "{}, Nb prev_dirs: {}".format(nb_s_input, nb_s_prev_dir))
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
