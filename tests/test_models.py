#!/usr/bin/env python
import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from dwi_ml.data.packed_sequences import (unpack_sequence,
                                          unpack_tensor_from_indices)
from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.models.embeddings_on_packed_sequences import keys_to_embeddings

from Learn2Track.models.learn2track_model import Learn2TrackModel
from Learn2Track.models.stacked_rnn import StackedRNN


def prepare_tensor(a):
    a = torch.as_tensor(a, dtype=torch.float32)
    return a


def create_batch():
    # input: shape (batch=2, seq=[3,2], input_size=4)
    nb_prev_dirs = 3

    print("\nCreating batch: 2 streamlines, the first has 3 timepoints "
          "and the second, 2. 4 features per point. nb_previous_dirs = {}"
          .format(nb_prev_dirs))

    # streamline 1: 4 timepoints = 3 directions
    flattened_dwi1 = np.array([[10, 11, 12, 13],
                               [50, 51, 52, 53],
                               [60, 62, 62, 63],
                               [140, 150, 160, 170]], dtype='float32')
    flattened_dwi1 = prepare_tensor(flattened_dwi1)
    directions1 = np.array([[1, 0, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype='float32')
    directions1 = prepare_tensor(directions1)

    # streamline 2: 3 timepoints = 2 directions
    flattened_dwi2 = np.array([[10, 11, 12, 13],
                               [20, 21, 22, 23],
                               [140, 150, 160, 170]], dtype='float32')
    flattened_dwi2 = prepare_tensor(flattened_dwi2)
    directions2 = np.array([[3, 0, 0],
                            [0, 3, 0]], dtype='float32')
    directions2 = prepare_tensor(directions2)

    # batch = 2 streamlines
    batch_x_data = [flattened_dwi1, flattened_dwi2]
    batch_directions = [directions1, directions2]

    # previous_dirs like in the dwi_ml's batch sampler
    # When previous dirs do not exist (ex, the 2nd previous dir at the first
    # time step), value is NaN.
    empty_coord = torch.zeros((1, 3), dtype=torch.float32)

    previous_dirs = \
        [torch.cat([torch.cat((empty_coord.repeat(min(len(s), i + 1), 1),
                               s[:-(i + 1)]))
                    for i in range(nb_prev_dirs)], dim=1)
         for s in batch_directions]

    print("-Previous dirs should be of size 2: {}".format(len(previous_dirs)))
    print("-Each tensor should be of size [nb_time_step, nb_previous_dir x 3] "
          "(the n previous dirs at each point).")
    print("  First tensor should be of size [3, 3x3]: {}"
          .format(previous_dirs[0].shape))
    print("  Second one should be of size [2, 3x3]: {}\n"
          .format(previous_dirs[1].shape))

    return batch_x_data, batch_directions, previous_dirs, nb_prev_dirs


def test_packing(streamlines):
    print("Input before packing: {}".format(streamlines))
    packed_sequence = pack_sequence(streamlines, enforce_sorted=False)
    inputs_tensor = packed_sequence.data
    print("Packed: {}".format(inputs_tensor))

    # Unpacking technique 1
    result = unpack_sequence(packed_sequence)
    print("Unpacked technique 1: {}".format(result))

    # Unpacking technique 2
    indices = unpack_sequence(packed_sequence, get_indices_only=True)
    result = unpack_tensor_from_indices(inputs_tensor, indices)
    print("Unpacked technique 2: {}".format(result))


def test_prev_dir_embedding(prev_directions, nb_prev_dirs):

    print('\nTesting identity embedding...')
    cls = keys_to_embeddings['no_embedding']
    model = cls(input_size=nb_prev_dirs * 3, output_size=nb_prev_dirs * 3)
    output = model(prev_directions)
    print('==> Should return itself. Output is: {}' .format(output))

    print('\nTesting neural network embedding, ...')
    cls = keys_to_embeddings['nn_embedding']
    model = cls(input_size=nb_prev_dirs * 3, output_size=8)
    output = model(prev_directions)
    print('==> Should return output of size 8. Result is: {}'
          .format(output))
    return model


def test_stacked_rnn(inputs):
    if type(inputs) == PackedSequence:
        input_size = inputs.data.shape[-1]
    else:
        input_size = inputs[0].shape[-1]
    print("Reminder. Input size is {}".format(input_size))

    print("\nTesting LSTM stacked_rnn...")
    model = StackedRNN('lstm', input_size, [5, 7],
                       use_skip_connections=True,
                       use_layer_normalization=True,
                       dropout=0.4)
    print("==> Model's computed output size should be 5+7=12: {}"
          .format(model.output_size))

    print('\nTesting forward when inputs is a {}...'.format(type(inputs)))
    model(inputs)
    print('...done')

    # output information printed during forward on debug logging
    return model


def test_learn2track(model_prev_dirs, inputs, prev_dirs):

    print('Using same prev dirs embeding model as previous test.')

    print('\nPreparing model...')
    input_size=4
    prev_dir_embedding_size = model_prev_dirs.output_size

    model = Learn2TrackModel(nb_previous_dirs, prev_dir_embedding_size,
                             'nn_embedding', input_size, 'nn_embedding', 1,
                             'lstm', [10, 12], use_skip_connection=True,
                             use_layer_normalization=True, dropout=0.5,
                             direction_getter_key='cosine-regression')

    print("\nUsing forward...")
    directions, state = model.forward(inputs, prev_dirs)

    print('\n==> Output should be of dim 3 (x,y,z): {}'.format(directions))

    return model, directions


def test_loss_functions(model: Learn2TrackModel, output, target_directions):

    if type(output) == PackedSequence:
        print("\nTesting loss function...")
        loss = model.compute_loss(output.data, target_directions.data)
        print("==> Loss: {}".format(loss))
    else:
        print("Loss function currently not working for list of tensors. "
              "You need to pack and use .data.")


if __name__ == '__main__':

    logging.basicConfig(level='INFO')

    (fake_input, fake_directions, fake_prev_dirs,
     nb_previous_dirs) = create_batch()

    print('****************************\n'
          'Testing packing and unpacking\n'
          '****************************\n')
    test_packing(fake_prev_dirs)

    # Preparing packed sequence data for the model
    # Hide the following lines to test on a list of tensors instead.
    # The loss can't be computed on lists in the direction getters for now.
    fake_prev_dirs = pack_sequence(fake_prev_dirs, enforce_sorted=False)
    fake_input = pack_sequence(fake_input, enforce_sorted=False)
    fake_directions = pack_sequence(fake_directions, enforce_sorted=False)

    print('\n****************************\n'
          'Testing previous dir embedding on packed sequences\n'
          '****************************\n')
    model_prev_dir = test_prev_dir_embedding(fake_prev_dirs, nb_previous_dirs)

    print('\n****************************\n'
          'Testing stacked RNN\n'
          '****************************\n')
    test_stacked_rnn(fake_input)

    print('\n****************************\n'
          'Testing Learn2track\n'
          '(prev_dir + stackedRNN + direction getter)\n'
          '****************************\n')
    (learn2track_model,
     out_directions) = test_learn2track(model_prev_dir, fake_input,
                                        fake_prev_dirs)

    print('\n****************************\n'
          'Testing loss function\n'
          '****************************\n')
    test_loss_functions(learn2track_model, out_directions, fake_directions)
