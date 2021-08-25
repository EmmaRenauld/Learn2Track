#!/usr/bin/env python
import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from dwi_ml.model.direction_getter_models import keys_to_direction_getters

from Learn2Track.model.embeddings import keys_to_embeddings
from Learn2Track.model.stacked_rnn import StackedRNN
from Learn2Track.model.learn2track_model import Learn2TrackModel
from Learn2Track.utils.packed_sequences import (unpack_sequence,
                                                unpack_tensor_from_indices)


def prepare_tensor(a):
    a = torch.as_tensor(a, dtype=torch.float32)
    return a


def create_batch():
    # input: shape (batch=2, seq=[3,2], input_size=4)
    nb_prev_dirs = 2

    print("\nCreating batch: 2 streamlines, the first has 3 timepoints "
          "and the second, 2. 4 features per point. nb_previous_dirs = {}"
          .format(nb_prev_dirs))

    # streamline 1: 3 timepoints
    flattened_dwi1 = np.array([[10, 11, 12, 13],
                               [50, 51, 52, 53],
                               [140, 150, 160, 170]], dtype='float32')
    flattened_dwi1 = prepare_tensor(flattened_dwi1)
    directions1 = np.array([[1, 0, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype='float32')
    directions1 = prepare_tensor(directions1)

    # streamline 2: 2 timepoints
    flattened_dwi2 = np.array([[10, 11, 12, 13],
                               [140, 150, 160, 170]], dtype='float32')
    flattened_dwi2 = prepare_tensor(flattened_dwi2)
    directions2 = np.array([[1, 0, 0],
                            [0, 1, 0]], dtype='float32')
    directions2 = prepare_tensor(directions2)

    # batch = 2 streamlines
    batch_x_data = [flattened_dwi1, flattened_dwi2]
    batch_directions = [directions1, directions2]

    # previous_dirs like in the dwi_ml's batch sampler
    # Size should be 2 x [nb_time_step, nb_previous_dir x 3]
    empty_coord = torch.zeros((1, 3), dtype=torch.float32) * float('NaN')
    previous_dirs = \
        [torch.cat([torch.cat((empty_coord.repeat(i + 1, 1),
                               s[:-(i + 1)]))
                    for i in range(nb_prev_dirs)], dim=1)
         for s in batch_directions]

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

    print('==> previous_dirs should be: \n'
          '     - nb_streamlines * size [nb_points_i * nb_prev_dirs*3]] = '
          '3x6 and 2x6 if it is a list, \n'
          '     - nb_points_total * nb_prev_dirs*3 = 5x6 if it is a packed '
          'sequence\n'
          '{}'
          .format(prev_directions))

    print('\nTesting identity embedding...')
    cls = keys_to_embeddings['no_embedding']
    model = cls(input_size=nb_prev_dirs * 3, output_size=nb_prev_dirs * 3)
    output = model(prev_directions)
    print('==> Should return itself. Output is:\n{}' .format(output))

    print('\nTesting neural network embedding, ...')
    cls = keys_to_embeddings['nn_embedding']
    model = cls(input_size=nb_prev_dirs * 3, output_size=8, nan_to_num=0)
    output = model(prev_directions)
    print('==> Should return output of size 8. Result is: \n'
          '{}'
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

    # output information printed during forward on debug logging
    return model


def test_learn2track(model_prev_dirs, inputs, prev_dirs):

    print('Using same prev dirs embeding model as previous test.')

    print('\nPreparing model...')
    # 1. embedding pf prev_dirs: using already instantiated
    # 2. embedding of input
    cls = keys_to_embeddings['no_embedding']
    model_input_embedding = cls(input_size=4, output_size=4)

    # 3. stacked_rnn
    input_embedding_size = model_input_embedding.output_size
    prev_dir_embedding_size = model_prev_dirs.output_size
    model_rnn = StackedRNN('lstm',
                           input_embedding_size + prev_dir_embedding_size,
                           [10, 12],
                           use_skip_connections=True,
                           use_layer_normalization=True,
                           dropout=0.4)

    # 4. Direction getter
    chosen_direction_getter = \
        keys_to_direction_getters['cosine-regression']
    model_direction = chosen_direction_getter(model_rnn.output_size)

    # 5. Final model.
    model = Learn2TrackModel(previous_dir_embedding_model=model_prev_dirs,
                             input_embedding_model=model_input_embedding,
                             rnn_model=model_rnn,
                             direction_getter_model=model_direction)

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
          'Testing previous dir embedding\n'
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
