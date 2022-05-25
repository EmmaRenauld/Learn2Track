# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from dwi_ml.data.dataset.single_subject_containers import SubjectDataAbstract
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.tracking.propagator import DWIMLPropagatorOneInput


class RecurrentPropagator(DWIMLPropagatorOneInput):
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
        model_uses_streamlines = True
        super().__init__(dataset, model, input_volume_group,
                         neighborhood_points, step_size, rk_order, algo, theta,
                         model_uses_streamlines, device)

        # Internal state:
        # - previous_dirs, already dealt with by super.
        # - For RNN: new parameter: The hidden states of the RNN
        self.hidden_recurrent_states = None

    def _reset_memory_state(self):
        super()._reset_memory_state()
        self.hidden_recurrent_states = None

    def _reverse_memory_state(self, line):
        """
        Prepare memory state for the backward pass. Anything your model needs
        to deal with in memory during streamline generation (ex, memory of the
        previous directions).

        Line: Already reversed line (streamline from the forward tracking).
        """
        super()._reverse_memory_state(line)

        # Reverse the streamlines
        logging.debug("Computing hidden RNN state at backward: recomputing "
                      "whole sequence to run model.")

        # Must re-run the model from scratch to get the hidden states
        # Either load all timepoints in memory and call model once.
        all_inputs = []

        # toDo Can be accelerated?
        for i in range(len(line)):
            all_inputs.append(self._prepare_inputs_at_pos(line[i]))

        # all_inputs is a list of n_points x tensor([1, nb_features])
        # creating a batch of 1 streamline with tensor[nb_points, nb_features]
        all_inputs = [torch.cat(all_inputs, dim=0)]

        # Running model
        _, self.hidden_recurrent_states = self.model(all_inputs, line,
                                                     self.device,
                                                     return_state=True)
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
        # Recurrent memory managed through the hidden state: no need to send
        # the whole streamline again.

        # Shape: [1, nb_features] --> only the current point, contrary to
        # during training, where the whole streamline is read and sent,
        # one-shot.
        inputs = self._prepare_inputs_at_pos(pos)

        logging.debug("Learn2track propagation step. Inputs: {}".format(inputs))

        # Sending [inputs] to simulate a batch to be packed.
        # Sending line's last 2 points, to compute one direction.
        model_outputs, hidden_states = self.model(
            [inputs], [torch.tensor(self.line[-2: -1])],
            self.hidden_recurrent_states, return_state=True)

        if get_state_only:
            return hidden_states
        else:
            return model_outputs
