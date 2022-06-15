# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.tracking.propagator import DWIMLPropagatorOneInput

logger = logging.getLogger('tracker_logger')


class RecurrentPropagator(DWIMLPropagatorOneInput):
    """
    To use a RNN for a generative process, the hidden recurrent states that
    would be passed (ex, h_(t-1), C_(t-1) for LSTM) need to be kept in memory
    as an additional input.

    In theory, the previous timepoints' inputs do not need to be kept, except
    for the backward tracking: the hidden recurrent states need to be computed
    from scratch. We will reload them all when starting backward, if necessary.
    """
    def __init__(self, dataset: MultisubjectSubset, subj_idx: int,
                 model: MainModelAbstract, input_volume_group: str,
                 step_size: float, rk_order: int,
                 algo: str, theta: float, device=None):
        model_uses_streamlines = True
        super().__init__(dataset, subj_idx, model, input_volume_group,
                         step_size, rk_order, algo, theta,
                         model_uses_streamlines, device)

        # Internal state:
        # - previous_dirs, already dealt with by super.
        # - For RNN: new parameter: The hidden states of the RNN
        self.hidden_recurrent_states = None

        if rk_order != 1:
            raise ValueError("Learn2track is not ready for runge-kutta "
                             "integration of order > 1.")

    def prepare_forward(self, seeding_pos):
        """
        Additionnally to usual preparation, we need to reset the recurrent
        hidden state.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)

        Returns
        -------
        tracking_info: None
            No initial tracking information necessary for the propagation.
        """
        logger.debug("Learn2track: Resetting propagator for new streamline.")
        self.hidden_recurrent_states = None

        return super().prepare_forward(seeding_pos)

    def prepare_backward(self, line, forward_dir):
        """
        Preparing backward. We need to recompute the hidden recurrent state
        for this half-streamline.

        Parameters
        ----------
        line: List
            Result from the forward tracking, reversed. Single line: list of
            coordinates. Simulatenous tracking: list of list of coordinates.
        forward_dir: ndarray (3,)
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,) or list[ndarray]
            Last direction of the streamline. If the streamline contains
            only the seeding point (forward tracking failed), simply inverse
            the forward direction.
        """
        logger.debug("Computing hidden RNN state at backward: run model on "
                     "(reversed) first half.")

        # Must re-run the model from scratch to get the hidden states
        # Either load all timepoints in memory and call model once.
        # Or loop.
        if isinstance(line[0], np.ndarray):  # List of coords = single tracking
            # Using the loop meant for multiple tracking to actually get
            # multiple positions
            all_inputs = [self._prepare_inputs_at_pos(line)]
            lines = [line]
        else:
            all_inputs = []
            for one_line in line:
                all_inputs.append(self._prepare_inputs_at_pos(one_line))
            lines = line

        # all_inputs is a list of
        # nb_streamlines x n_points x tensor([1, nb_features])
        # creating a batch of streamlines as tensor[nb_points, nb_features]
        all_inputs = [torch.cat(streamline_inputs, dim=0)
                      for streamline_inputs in all_inputs]

        # Running model. If we send is_tracking=True, will only compute the
        # previous dirs for the last point. To mimic training, we have to
        # add an additional fake point to the streamline, not used.
        lines = [torch.cat((torch.tensor(line),
                            torch.zeros(1, 3)), dim=0)
                 for line in lines]

        # Also, warning: creating a tensor from a list of np arrays is low.
        _, self.hidden_recurrent_states = self.model(
            all_inputs, lines, is_tracking=False, return_state=True)

        return super().prepare_backward(line, forward_dir)

    def _get_model_outputs_at_pos(self, n_pos):
        """
        Overriding dwi_ml: model needs to use the hidden recurrent states +
        we need to pack the data.

        Parameters
        ----------
        n_pos: list of ndarrays
            Current position coordinates for each streamline.
        """
        # Copying the beginning of super's method
        inputs = self._prepare_inputs_at_pos(n_pos)

        # In super, they add an additional point to mimic training. Here we
        # have already managed it in the forward by sending is_tracking.
        # Converting lines to tensors
        lines = [torch.tensor(np.vstack(line)) for line in self.current_lines]

        # For RNN however, we need to send the hidden state too.
        model_outputs, hidden_states = self.model(
            inputs, lines, self.hidden_recurrent_states,
            return_state=True, is_tracking=True)

        self.hidden_recurrent_states = hidden_states
        return model_outputs
