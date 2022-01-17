# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from dwi_ml.data.dataset.single_subject_containers import SubjectDataAbstract
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.tracking.propagator import DWIMLPropagatorOneInputAndPD


class RecurrentPropagator(DWIMLPropagatorOneInputAndPD):
    """
    To use a RNN for a generative process, the hidden recurrent states that
    would be passed (ex, h_(t-1), C_(t-1) for LSTM) need to be kept in memory
    as an additional input.

    In theory, the previous timepoints' inputs do not need to be kept, except
    for the backward tracking: the hidden recurrent states need to be computed
    from scratch. We will reload them all when starting backward, if necessary.
    """
    def __init__(self, dataset: SubjectDataAbstract, model: MainModelAbstract,
                 input_volume_group: str, neighborhood_points: np.ndarray,
                 step_size: float, rk_order: int, algo: str, theta: float,
                 device=None):
        super().__init__(dataset, model, input_volume_group,
                         neighborhood_points, step_size, rk_order, algo, theta,
                         device)

        # Internal state:
        # - previous_dirs, already dealt with by super.
        # - For RNN: new parameter: The hidden states of the RNN
        self.hidden_recurrent_states = None

    def _reset_memory_state(self):
        super()._reset_memory_state()
        self.hidden_recurrent_states = None

    def _reverse_memory_state(self, line):
        # Reverse the streamlines
        super()._reverse_memory_state(line)

        logging.debug("Computing hidden RNN state at backward: recomputing "
                      "whole sequence to run model.")

        # Must re-run the model from scratch to get the hidden states
        # Either load all timepoints in memory and call model once.
        all_inputs = []
        all_n_previous_dirs = []
        for i in range(len(line)):
            # toDo
            #  We could also prepare inputs only, and previous dirs out of loop
            inputs, n_previous_dirs = self._prepare_inputs_at_pos(line[i])
            all_inputs.append(inputs)
            all_n_previous_dirs.append(n_previous_dirs[0])

        # all_inputs is a lit of n_points x tensor([1, nb_features])
        # creating a batch of 1 streamline with tensor[nb_points, nb_features]
        all_inputs = [torch.cat(all_inputs, dim=0)]

        # all_previous dirs n_points x tensor([1, nb_prev_dirs * 3])
        # creating a batch of 1 streamline with tensor[nb_points, nb_prev * 3]
        all_n_previous_dirs = [torch.cat(all_n_previous_dirs, dim=0)]

        # Running model
        _, self.hidden_recurrent_states = self.model(all_inputs,
                                                     all_n_previous_dirs,
                                                     self.device)
        logging.debug("Done.")

    def _update_state_after_propagation_step(self, new_pos, new_dir):
        # First re-run the model with final new_dir to get final hidden states
        # (ex; with Runge-Kutta integration, we wouldn't know which hidden
        # state to use, so we need to recompute)
        # toDo. With runge-kutta=1, we could get the hidden states right away.
        self.hidden_recurrent_states = self._get_model_outputs_at_pos(
            new_pos, get_state_only=True)

        # Only then update previous dirs
        super()._update_state_after_propagation_step(new_pos, new_dir)

    def _get_model_outputs_at_pos(self, pos, get_state_only=False):
        """
        Overriding dwi_ml: model needs to use the hidden recurrent states +
        we need to pack the data.

        Parameters
        ----------
        pos: ndarray (3,)
            Current position coordinates.
        get_state_only: bool
            If true, returns the hidden RNN states instead of the model output.
        """
        # Sending [inputs] simulate a batch of inputs to be packed.
        # Shape: [1, nb_features] --> only the current point, contrary to
        # during training, where the whole streamline is read and sent,
        # one-shot.
        inputs, n_prev_dirs = self._prepare_inputs_at_pos(pos)

        # Sending inputs to lists to simulate a batch to be packed.
        # n_previous_dirs is already a list of 1.
        inputs = [inputs]
        model_outputs, hidden_states = self.model(inputs, n_prev_dirs,
                                                  self.device,
                                                  self.hidden_recurrent_states)

        if get_state_only:
            return hidden_states
        else:
            return model_outputs
