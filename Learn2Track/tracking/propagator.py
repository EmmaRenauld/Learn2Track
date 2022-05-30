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
            Result from the forward tracking, already reversed.
        forward_dir: ndarray (3,)
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,)
            Last direction of the streamline. If the streamline contains
            only the seeding point (forward tracking failed), simply inverse
            the forward direction.
        """
        logger.debug("Computing hidden RNN state at backward: run model on "
                     "(reversed) first half.")

        # Must re-run the model from scratch to get the hidden states
        # Either load all timepoints in memory and call model once.
        # Or loop.
        all_inputs = []
        for i in range(len(line)):
            all_inputs.append(self._prepare_inputs_at_pos(line[i]))

        # all_inputs is a list of n_points x tensor([1, nb_features])
        # creating a batch of 1 streamline with tensor[nb_points, nb_features]
        all_inputs = torch.cat(all_inputs, dim=0)

        # Running model. If we send is_tracking=True, will only compute the
        # previous dirs for the last point. To mimic training, we have to
        # add an additional fake point to the streamline, not used.
        fake_line = line.copy()
        fake_line.append(np.asarray([0., 0., 0.]))

        # Also, warning: creating a tensor from a list of np arrays is low.
        fake_line = torch.tensor(np.vstack(fake_line))
        _, self.hidden_recurrent_states = self.model(
            [all_inputs], [fake_line], return_state=True)
        logger.debug("Done.")
        return super().prepare_backward(line, forward_dir)

    def _get_model_outputs_at_pos(self, pos):
        """
        Overriding dwi_ml: model needs to use the hidden recurrent states +
        we need to pack the data.

        Parameters
        ----------
        pos: ndarray (3,)
            Current position coordinates.
        """
        # Recurrent memory managed through the hidden state: no need to send
        # the whole streamline again.

        # Shape: [1, nb_features] --> only the current point, contrary to
        # during training, where the whole streamline is read and sent,
        # one-shot.
        inputs = self._prepare_inputs_at_pos(pos)

        # Sending [inputs], [line] to simulate a batch to be packed.
        # During training, we have one more point then the number of
        # inputs: the last point is only used to get the direction.
        # Adding a fake last point that won't be used.
        line = torch.tensor(np.vstack(self.line))

        model_outputs, hidden_states = self.model(
            [inputs], [line], self.hidden_recurrent_states, return_state=True,
            is_tracking=True)

        self.hidden_recurrent_states = hidden_states
        return model_outputs
