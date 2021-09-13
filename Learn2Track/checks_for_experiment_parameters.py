# -*- coding: utf-8 -*-
import logging
from typing import List

""" Various arg_parser equivalents."""

from dwi_ml.experiment.checks_for_experiment_parameters import (
    check_bool_instance_not_none, check_float_or_none_instance,
    check_int_or_none_instance, check_str_or_none_instance,
    check_all_experiment_parameters as dwi_ml_checks)
from dwi_ml.model.direction_getter_models import keys_to_direction_getters

from Learn2Track.model.embeddings import keys_to_embeddings
from Learn2Track.model.stacked_rnn import keys_to_rnn_class


def check_clip_grad(clip_grad: bool):
    check_bool_instance_not_none(clip_grad, 'clip_grad')

    return clip_grad


def check_prev_dirs_embedding(embedding_key: str, nb_previous_dirs: int,
                              output_size: int):
    check_str_or_none_instance(embedding_key, 'previous_dir embedding key')
    check_int_or_none_instance(output_size, 'prev dir embedding output size')

    # Allowing None for embedding_key
    if embedding_key is None:
        if nb_previous_dirs > 0:
            raise ValueError("Previous dir embedding cannot be none when you "
                             "set nb_previous_dirs>0. You probably meant "
                             "'no_embedding'?")
        if output_size:
            logging.warning(
                "You have set the embedding key to None, suggesting you will "
                "not include any previous dirs, but the previous_dir "
                "embedding output size is not None. Ignoring.")
    elif embedding_key not in keys_to_embeddings:
        raise ValueError("Previous dir embedding key is not one of the "
                         "allowed values: {}"
                         .format(keys_to_embeddings.keys()))

    return keys_to_embeddings[embedding_key], output_size


def check_inputs_embedding(embedding_key: str):
    check_str_or_none_instance(embedding_key, 'input embedding key')

    # NOT Allowing None for embedding_key
    if embedding_key not in keys_to_embeddings:
        raise ValueError("Inputs embedding key is not one of the allowed "
                         "values: {}".format(keys_to_embeddings.keys()))

    return keys_to_embeddings[embedding_key]


def check_rnn_params(rnn_key: str, layer_sizes: List[int]):
    # 1) RNN key
    check_str_or_none_instance(rnn_key, 'rnn_key')
    if rnn_key not in keys_to_rnn_class:
        raise ValueError("RNN key is not one of the allowed values: {}"
                         .format(keys_to_rnn_class.keys()))

    # 2) layer_sizes
    if not isinstance(layer_sizes, list):
        raise ValueError("RNN layer sizes should be a list!")
    if not isinstance(layer_sizes[0], int):
        raise ValueError("RNN layers should contain integers!")

    return rnn_key, layer_sizes


def check_direction_getter(key: str):
    check_str_or_none_instance(key, 'direction getter key')

    # NOT Allowing None for rnn_key
    if key not in keys_to_direction_getters:
        raise ValueError("Direction getter key is not one of the allowed "
                         "values: {}".format(keys_to_direction_getters.keys()))
    return keys_to_direction_getters[key]


def check_all_experiment_parameters(conf: dict):
    (sampling_params, training_params, model_params,
     memory_params, randomization) = dwi_ml_checks(conf)

    # training additional params:
    training_params.update({
        'clip_grad': check_bool_instance_not_none(
            conf['training']['clip_grad'], 'clip_grad')})

    # model additional params:
    cp, sp = check_prev_dirs_embedding(
        conf['model']['previous_dirs']['embedding'],
        model_params['nb_previous_dirs'],
        conf['model']['previous_dirs']['embedding_output_size'])
    rk, rls = check_rnn_params(conf['model']['rnn']['rnn_key'],
                               conf['model']['rnn']['layer_sizes'])
    model_params.update({
        'prev_dirs_embedding_cls': cp,
        'prev_dirs_embedding_size': sp,
        'input_embedding_cls': check_inputs_embedding(
            conf['model']['input']['embedding']),
        'input_embedding_size_ratio': check_float_or_none_instance(
            conf['model']['input']['output_size_ratio'],
            'input embedding size'),
        'rnn_key': rk,
        'rnn_layer_sizes': rls,
        'dropout': check_float_or_none_instance(
            conf['model']['rnn']['dropout'], 'dropout'),
        'direction_getter_cls': check_direction_getter(
            conf['model']['direction_getter']['key'])})

    return (sampling_params, training_params, model_params, memory_params,
            randomization)
