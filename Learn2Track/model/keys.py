# -*- coding: utf-8 -*-

import torch

from Learn2Track.model.embeddings import (NNEmbedding,
                                          NoEmbedding)
# See also dwi_ml.model.direction_getter_models.KEY_TO_DIRECTION_GETTER_MODEL


KEY_TO_RNN_CLASS = {'lstm': torch.nn.LSTM,
                    'gru': torch.nn.GRU}

KEY_TO_PREV_DIR_EMBEDDING = {'no_embedding': NoEmbedding,
                             'nn': NNEmbedding}
