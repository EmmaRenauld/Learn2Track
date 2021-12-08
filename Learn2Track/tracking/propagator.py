# -*- coding: utf-8 -*-
from dwi_ml.tracking.propagator import DWIMLPropagatorOneInputAndPD

from Learn2Track.data_loaders.tracking_field import RecurrentTrackingField


class RecurrentPropagator(DWIMLPropagatorOneInputAndPD):
    """
    Associated to a recurrent model with inputs:
        - inputs: a volume from the hdf5 + (neighborhood)
        - previous_dirs: the n previous dirs.

    The difference with dwi_ml is that the model outputs its hidden recurrent
    states, that need to be kept in memory between steps of the propagation.
    (ex, h_(t-1), C_(t-1) for LSTM)
    """

    def __init__(self, tracking_field: RecurrentTrackingField,
                 step_size: float, rk_order: int, algo: str, theta: float):
        super().__init__(tracking_field, step_size, rk_order, algo, theta)

        # New parameters: The hidden states of the RNN.
        self.hidden_recurrent_states = None

        # The hidden states will be updated at the first call of the model
        # during each propagation step (with Runge-Kutta integration, the model
        # is ran more than once per step).
        self.save_next_state = True

    def initialize(self, pos, track_forward_only):
        self.save_next_state = True
        return super().initialize(pos, track_forward_only)

    def start_backward(self):
        """
        Will need to be called between forward and backward tracking.
        """
        self.previous_dirs = self.previous_dirs.reverse()
        self.hidden_recurrent_states = None
        self.save_next_state = True

    def _get_model_outputs(self, pos):
        model_outputs, next_states = \
            self.tracking_field.get_model_outputs_at_pos(
                pos, self.previous_dirs, self.hidden_recurrent_states)

        if self.save_next_state:
            # As long as propagation is not officially done, next calls to
            # this method will be during the runge-kutta integration and
            # states should not be recorded.
            self.hidden_recurrent_states = next_states
            self.save_next_state = False

        return model_outputs

    def propagate(self, pos, v_in):
        new_pos, new_dir, is_direction_valid = super().propagate(pos, v_in)
        self.save_next_state = True

        return new_pos, new_dir, is_direction_valid
