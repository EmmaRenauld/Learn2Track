# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.embeddings_on_packed_sequences import keys_to_embeddings
from dwi_ml.models.direction_getter_models import keys_to_direction_getters

from Learn2Track.models.learn2track_model import Learn2TrackModel


def add_model_args(p: argparse.ArgumentParser):
    prev_dirs_g = p.add_argument_group(
        "Learn2track model: Previous directions embedding layer")
    prev_dirs_g.add_argument(
        '--nb_previous_dirs', type=int, default=0, metavar='n',
        help="Concatenate the n previous streamline directions to the input "
             "vector. \nDefault: 0")
    prev_dirs_g.add_argument(
        '--prev_dirs_embedding_key', choices=keys_to_embeddings.keys(),
        default='no_embedding',
        help="Type of model for the previous directions embedding layer.\n"
             "Default: no_embedding (identify model).")
    prev_dirs_g.add_argument(
        '--prev_dirs_embedding_size', type=int, metavar='s',
        help="Size of the output after passing the previous dirs through the "
             "embedding \nlayer.")

    inputs_g = p.add_argument_group(
        "Learn2track model: Main inputs embedding layer")
    inputs_g.add_argument(
        '--input_embedding_key', choices=keys_to_embeddings.keys(),
        default='no_embedding',
        help="Type of model for the inputs embedding layer.\n"
             "Default: no_embedding (identify model).")
    em_size = inputs_g.add_mutually_exclusive_group()
    em_size.add_argument(
        '--input_embedding_size', type=float, metavar='s',
        help="Size of the output after passing the previous dirs through the "
             "embedding layer.")
    em_size.add_argument(
        '--input_embedding_size_ratio', type=float, metavar='r',
        help="Size of the output after passing the previous dirs through the "
             "embedding layer. \nThe inputs size (i.e. number of features per "
             "voxel) will be verified when \nloading the data, and the "
             "embedding size wil be output_size_ratio*nb_features.\n"
             "Default: 1.")

    rnn_g = p.add_argument_group("Learn2track model: RNN layer")
    rnn_g.add_argument(
        '--rnn_key', choices=['lstm', 'gru'], default='lstm',
        help="RNN version. Default: lstm.")
    rnn_g.add_argument(
        '--rnn_layer_sizes', type=int, nargs='+', metavar='s',
        default=[100, 100],
        help="The output size after each layer (the real output size depends "
             "on skip \nconnections). Default = [100, 100]")
    rnn_g.add_argument(
        '--dropout', type=float, default=0., metavar='r',
        help="Dropout ratio. If >0, add a dropout layer between RNN layers.")
    rnn_g.add_argument(
        '--use_skip_connection', action='store_true',
        help="Add skip connections. The pattern for skip connections is as "
             "seen here:\n"
             "https://arxiv.org/pdf/1308.0850v5.pdf")
    rnn_g.add_argument(
        '--use_layer_normalization', action='store_true',
        help="Add layer normalization. Explained here: \n"
             "https://arxiv.org/pdf/1607.06450.pdf"
    )

    dg_g = p.add_argument_group("Learn2track model: Direction getter layer")
    dg_g.add_argument(
        '--direction_getter_key', choices=keys_to_direction_getters.keys(),
        help="Model for the direction getter layer."
    )

    g = p.add_argument_group("Learn2track model: Prepare targets")
    g.add_argument(
        '--normalize_directions', action='store_true',
        help="If true, directions will be normalized. If the step size is "
             "fixed, it shouldn't \nmake any difference. If streamlines are "
             "compressed, in theory you should normalize, \nbut you could "
             "hope that not normalizing could give back to the algorithm a \n"
             "sense of distance between points.")


def prepare_model(args, input_size):
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        if args.nb_previous_dirs == 0:
            args.prev_dirs_embedding_key = None
        if args.prev_dirs_embedding_key == 'no_embedding':
            if args.prev_dirs_embedding_size != 3 * args.nb_previous_dirs:
                logging.warning("With identity embedding (no_embedding), "
                                "output_size must equal input_size. User "
                                "value {} discarted and new output_size value "
                                "will be: {}"
                                .format(args.prev_dirs_embedding_size,
                                        3 * args.nb_previous_dirs))
                args.prev_dirs_embedding_size = 3 * args.nb_previous_dirs
        elif args.prev_dirs_embedding_size is None:
            raise ValueError("You must provide the "
                             "prev_dirs_embedding_size.")

        input_embedding_size = None
        if args.input_embedding_size:
            input_embedding_size = args.input_embedding_size
        elif args.input_embedding_size_ratio:
            input_embedding_size = \
                int(input_size * args.input_embedding_size_ratio)
        if args.input_embedding_key == 'no_embedding':
            if input_embedding_size != input_size:
                logging.warning("With identity embedding (no_embedding), "
                                "output_size must equal input_size. User "
                                "value {} discarted and new output_size value "
                                "will be: {}"
                                .format(input_embedding_size, input_size))
                input_embedding_size = input_size
        elif input_embedding_size is None:
            raise ValueError("You must provide the input_embedding_size or "
                             "the input_embedding_size_ratio.")

        model = Learn2TrackModel(
            args.experiment_name,
            # PREVIOUS DIRS
            prev_dirs_embedding_key=args.prev_dirs_embedding_key,
            prev_dirs_embedding_size=args.prev_dirs_embedding_size,
            nb_previous_dirs=args.nb_previous_dirs,
            # INPUTS
            input_embedding_key=args.input_embedding_key,
            input_embedding_size=input_embedding_size,
            nb_features=input_size,
            # RNN
            rnn_key=args.rnn_key, rnn_layer_sizes=args.rnn_layer_sizes,
            dropout=args.dropout,
            use_layer_normalization=args.use_layer_normalization,
            use_skip_connection=args.use_skip_connection,
            # DIRECTION GETTER
            direction_getter_key=args.direction_getter_key,
            # Other
            normalize_directions=args.normalize_directions)

        # logging.info("Learn2track model user-defined parameters: \n" +
        #              format_dict_to_str(model.params) + '\n')
        logging.info("Learn2track model final parameters:" +
                     format_dict_to_str(model.params_per_layer))
    return model
