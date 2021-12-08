# -*- coding: utf-8 -*-
import numpy as np
import torch
from dwi_ml.data.dataset.single_subject_containers import SubjectDataAbstract
from dwi_ml.data_loaders.tracking_field import DWIMLTrackingFieldOneInputAndPD
from torch.nn.utils.rnn import pack_sequence

from Learn2Track.models.learn2track_model import Learn2TrackModel


class RecurrentTrackingField(DWIMLTrackingFieldOneInputAndPD):
    """
    To use a RNN for a generative process, the hidden recurrent states that
    would be passed (ex, h_(t-1), C_(t-1) for LSTM) need to be kept in memory
    as an additional input.
    """
    def __init__(self, model: Learn2TrackModel,
                 subj_data: SubjectDataAbstract, input_volume_group: str,
                 neighborhood_type, neighborhood_radius):
        super().__init__(model, subj_data, input_volume_group,
                         neighborhood_type, neighborhood_radius)

    def get_model_outputs_at_pos(self, pos, previous_dirs=None,
                                 hidden_recurrent_state=None):
        """
        Overriding dwi_ml: model needs to use the hidden recurrent states.

        Parameters
        ----------
        pos: ndarray (3,)
            Current position coordinates.
        previous_dirs: 3D coordinates
            List of the previous directions
            Size: (length of the streamline - 1) x 3. If this is the first step
            of the streamline, use None.
        hidden_recurrent_state: states
            The last step hidden states (h_(t-1), C_(t-1) for LSTM) for each
            layer.
        """
        inputs = self.subj_data.mri_data_list[self.volume_group]

        # Get pos in voxel world
        pos_vox = inputs.as_data_volume.voxmm_to_vox(*pos, self.origin)

        # torch trilinear interpolation uses origin='corner'
        if self.origin == 'center':
            pos_vox += 0.5

        inputs_arranged, _, = self.model.prepare_inputs(
            inputs.as_tensor, np.asarray([pos_vox]), self.device)
        if previous_dirs is not None and len(previous_dirs) > 0:
            previous_dirs = [torch.tensor(np.asarray(previous_dirs))]

        print("PREVIOUS DIRS: {}".format(previous_dirs))

        n_previous_dirs = self.model.prepare_previous_dirs(
            previous_dirs, self.device)

        print("PREVIOUS DIRS FORMATTED: {}".format(n_previous_dirs))

        # Packing data
        inputs_packed = pack_sequence([inputs_arranged], enforce_sorted=False)
        if len(n_previous_dirs) > 0:
            n_previous_dirs = pack_sequence(n_previous_dirs,
                                            enforce_sorted=False)
        else:
            n_previous_dirs = None

        model_outputs, new_states = self.model.forward(
            inputs_packed, n_previous_dirs, hidden_recurrent_state)

        return model_outputs, new_states
