"""Fetch paths according to different directory structures."""
from pathlib import Path

from Learn2Track.data.definitions import (DirStructure, _BUNDLES_DIR_FLAT,
                                          _BUNDLES_DIR_TRACTOFLOW,
                                          _DATA_2_FILENAME_FLAT,
                                          _DATA_2_FILEPATH_TRACTOFLOW,
                                          _FILE_SPLIT_TOKEN_FLAT,
                                          _FILE_SPLIT_TOKEN_TRACTOFLOW)


def fetch_data_path(subject_id: str, s_subject_path: str, data_choice: str,
                    dir_structure: DirStructure):
    """

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    s_subject_path : str
        Path to subject folder.
    data_choice : str
        Choice of data to include in the datasets.
    dir_structure : DirStructure
        Directory structure to fetch from.

    Returns
    -------
    final_path : str
        Final path to the requested data

    """
    try:
        if dir_structure == DirStructure.FLAT:
            filename = _DATA_2_FILENAME_FLAT[data_choice]
            return str(
                Path(s_subject_path).joinpath(
                    "{}_{}".format(subject_id, filename)))

        elif dir_structure == DirStructure.TRACTOFLOW:
            folder, filename = _DATA_2_FILEPATH_TRACTOFLOW[data_choice]
            return str(Path(s_subject_path).joinpath(folder).joinpath(
                "{}__{}".format(subject_id, filename)))
    except KeyError as e:
        raise ValueError("Unrecognized data choice: {}".format(data_choice))

    raise ValueError(
        "Unrecognized directory structure: {}".format(dir_structure))


def fetch_bundles_paths(subject_id: str, s_subject_path: str,
                        dir_structure: DirStructure):
    """

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    s_subject_path : str
        Path to subject folder.
    dir_structure : DirStructure
        Directory structure to fetch from.

    Returns
    -------
    bundles_dict : dict of (str, list(str))
        Dictionary matching each available bundle name/id to its file path.
    """
    if dir_structure == DirStructure.FLAT:
        bundles_dir = _BUNDLES_DIR_FLAT
        split_token = _FILE_SPLIT_TOKEN_FLAT
    elif dir_structure == DirStructure.TRACTOFLOW:
        bundles_dir = _BUNDLES_DIR_TRACTOFLOW
        split_token = _FILE_SPLIT_TOKEN_TRACTOFLOW
    else:
        raise ValueError(
            "Unrecognized directory structure: {}".format(dir_structure))

    # Get all available bundles in folder
    p_bundles_paths = Path(s_subject_path).joinpath(bundles_dir).glob('*.trk')

    # Fill dict with [bundle_name -> bundle_path]
    bundles_dict = {}
    for p_bundle_path in p_bundles_paths:
        bundle_name = _extract_bundle_name(p_bundle_path.name, subject_id,
                                           split_token)
        bundles_dict[bundle_name] = str(p_bundle_path)


def _extract_bundle_name(bundle_filename, subject_id, split_token):
    """

    Parameters
    ----------
    bundle_filename : str
        Bundle filename.
    subject_id : str
        Subject identifier.
    split_token : str
        Token splitting subject id from bundle name in filename.

    Returns
    -------
    bundle_name : str
        Name/identifier of the bundle.
    """
    return bundle_filename.lstrip(subject_id + split_token).rstrip('.trk')
