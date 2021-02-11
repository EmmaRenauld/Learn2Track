from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np


def is_tractogram_in_same_space(t1: StatefulTractogram,
                                t2: StatefulTractogram) -> bool:
    """Validates if two tractograms have the same space reference.

    Parameters
    ----------
    t1 : StatefulTractogram
        First tractogram.
    t2 : StatefulTractogram
        Second tractogram.

    Returns
    -------
    is_valid : bool
        True if the two tractograms are in the same space.
    """
    is_valid = True
    tested_attributes = map((lambda x: x[0] == x[1]),
                            zip(t1.space_attributes, t2.space_attributes))
    for attribute_array in tested_attributes:
        is_valid = is_valid and np.all(attribute_array)
    return is_valid
