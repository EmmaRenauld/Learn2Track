# -*- coding: utf-8 -*-
from typing import List, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import (PackedSequence, pack_sequence)

from dwi_ml.model.main_models import ModelAbstract

from Learn2Track.packed_sequences import (unpack_sequence,
                                          unpack_tensor_from_indices)

"""
Data needs to be usable as torch tensors or as packed sequences.
"""


class EmbeddingAbstract(ModelAbstract):
    def __init__(self, input_size: int, output_size: int = 128):
        """
        Params
        -------
        input_size: int
            Size of input data or, in the case of a sequence, of each data
            point. If working with streamlines, probably a multiple of 3
            ([x,y,z] for each direction).
        output_size: int
            Size of output data or of each output data point.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    @property
    def attributes(self):
        # We need real int types, not numpy.int64, not recognized by json
        # dumps.
        attributes = {
            'input_size': int(self.input_size),
            'output_size': int(self.output_size),
        }
        return attributes

    @property
    def hyperparameters(self):
        return {}

    def forward(self, inputs: Union[Tensor, List[Tensor], PackedSequence]):
        """
        From tensor to tensor
        or
        From list of tensors (ex, for each streamline) to a list or tensors
          (can't simply merge them if streamlines don't have all the same
           length. Packing sequence, using merged tensor, calling forward,
           unpacking.)
        or
        From PackedSequence to PackedSequence
        """
        raise NotImplementedError


class NNEmbedding(EmbeddingAbstract):
    def __init__(self, input_size, output_size: int):
        """
        Params
        ------
        intput_size: int
            See super.
        output_size: int
            See super. Default for the NN case: Philippe had set 128 in version
            1 of learn2track. Rationale?
        """
        super().__init__(input_size, output_size)
        self.linear = torch.nn.Linear(self.input_size, self.output_size)
        self.relu = torch.nn.ReLU()

    @property
    def attributes(self):
        attrs = super().attributes  # type: dict
        attrs.update({
            'key': 'nn_embedding'
        })
        return attrs

    def forward(self, inputs: Union[Tensor, List[Tensor], PackedSequence]):
        """See super."""
        self.log.debug("Embedding: running Neural networks' forward")
        packed_sequence = []
        if isinstance(inputs, list):
            if isinstance(inputs[0], torch.Tensor):
                nb_s = len(inputs)
                self.log.debug("input is a list of {} tensors (probably "
                               "corresponding to the number of streamlines). "
                               "Packing, we will unpack later.".format(nb_s))
                packed_sequence = pack_sequence(inputs, enforce_sorted=False)
                inputs_tensor = packed_sequence.data
            else:
                raise ValueError("Input must be a tensor or a list of "
                                 "tensors.")
        elif isinstance(inputs, torch.Tensor):
            inputs_tensor = inputs
        elif isinstance(inputs, PackedSequence):
            inputs_tensor = inputs.data
        else:
            raise ValueError("Input must be a tensor or a list of tensors.")

        # Calling forward.
        result = self.linear(inputs_tensor)
        result = self.relu(result)

        if isinstance(inputs, list):
            self.log.debug("Sending packed_data back to list.")
            # The total number of timepoints should have changed.
            assert packed_sequence.data.shape[0] == result.shape[0]

            # Number of features per data point has changed: can't unpack
            # directly.
            indices = unpack_sequence(packed_sequence, get_indices_only=True)
            result = unpack_tensor_from_indices(result, indices)
        elif isinstance(inputs, PackedSequence):
            self.log.debug("Packing results")
            result = PackedSequence(result, inputs.batch_sizes,
                                    inputs.sorted_indices,
                                    inputs.unsorted_indices)
        return result


class NoEmbedding(EmbeddingAbstract):
    def __init__(self, input_size, output_size: int = None):
        if output_size is None:
            output_size = input_size
        if input_size != output_size:
            self.log.debug("Identity embedding should have input_size == "
                           "output_size. Not stopping now but this won't work "
                           "if your data does not follow the shape you are "
                           "suggesting.")

        super().__init__(input_size, output_size)
        self.identity = torch.nn.Identity()

    def forward(self, inputs=None):
        self.log.debug("Embedding: running identity's forward")
        # toDo. Should check that input size = self.input_size but we don't
        #  know how the data is organized.
        result = self.identity(inputs)
        return result

    @property
    def attributes(self):
        attrs = super().attributes  # type: dict
        attrs.update({
            'key': 'no_embedding'
        })
        return attrs


class CNNEmbedding(EmbeddingAbstract):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.cnn_layer = torch.nn.Conv3d

    @property
    def attributes(self):
        params = super().attributes  # type: dict
        other_parameters = {
            'layers': 'non-defined-yet',
            'key': 'cnn_embedding'
        }
        return params.update(other_parameters)

    def forward(self, inputs: Union[Tensor, List[Tensor], PackedSequence]):
        raise NotImplementedError


keys_to_embeddings = {'no_embedding': NoEmbedding,
                      'nn_embedding': NNEmbedding,
                      'cnn_embedding': CNNEmbedding}
