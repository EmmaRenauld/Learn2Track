#!/usr/bin/env python
import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence

from dwi_ml.data.packed_sequences import (unpack_sequence,
                                          unpack_tensor_from_indices)


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


def main():
    logging.basicConfig(level='INFO')

    empty_coord = torch.zeros((1, 3), dtype=torch.float32)

    directions1 = torch.tensor(np.array([[1, 0, 0],
                                         [1, 1, 1],
                                         [0, 1, 0]], dtype='float32'))
    directions2 = torch.tensor(np.array([[2, 0, 0],
                                         [2, 2, 2],
                                         [0, 2, 0]], dtype='float32'))
    streamlines = [directions1, directions2]

    nb_prev_dirs = 4
    previous_dirs = \
        [torch.cat([torch.cat((empty_coord.repeat(min(len(s), i + 1), 1),
                               s[:-(i + 1)]))
                    for i in range(nb_prev_dirs)], dim=1)
         for s in streamlines]

    print('****************************\n'
          'Testing packing and unpacking\n'
          '****************************\n')
    test_packing(previous_dirs)


if __name__ == '__main__':
    main()
