#!/usr/bin/env python
"""Create an HDF5 dataset from a TractoFLow BIDS directory structure."""
import argparse
import logging
import os
from pathlib import Path
import pdb
import sys
from typing import List

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import length
import h5py
import nibabel
import numpy as np

from Learn2Track.data.definitions import DirStructure
from Learn2Track.data.fetcher import fetch_bundles_paths, fetch_data_path
from Learn2Track.data.validation import is_tractogram_in_same_space
from Learn2Track.utils.monitoring import Timer


def _parse_args():
    """Argument parsing."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('path', help="Path to the main dataset TractoFlow BIDS "
                                "directory.")

    # Directory structure
    p.add_argument('--dir-structure', type=DirStructure.from_string,
                   choices=list(DirStructure), default=DirStructure.FLAT,
                   help="Directory structure. If flat, root dir contains one "
                        "folder per subject, which contains all files prefixed "
                        "with subject id. [%(default)s]")

    # Data representation choices
    p.add_argument('--data-repr', required=True,
                   choices=["dwi", "dwi_odf", "fodf", "fodf_peaks"],
                   nargs='+',
                   default=["dwi_odf"],
                   help="Choice of data representation to include in the "
                        "dataset")

    # Streamlines
    p.add_argument('--streamlines', action="store_true",
                   help="If true, include all streamlines in the dataset")
    p.add_argument('--ref', choices=["dwi", "dwi_odf", "fodf", "fodf_peaks"],
                   help="Required if --streamlines is provided. Defines"
                        "which data to use as reference when sending "
                        "streamlines to VOX space (In case it is different "
                        "from the space used for tracking).")

    # Other facultative arguments
    p.add_argument('--name', type=str, help="Dataset name [Default uses "
                                            "date and time of processing].")

    p.add_argument('--logging', type=str,
                   choices=['error', 'warning', 'info', 'debug'],
                   default='warning',
                   help="Choose logging level. [warning]")

    p.add_argument('-f', '--force', action='store_true',
                   help="If set, overwrite existing dataset.")

    args = p.parse_args()
    return args


def main():
    """Parse arguments, generate hdf5 dataset and save it on disk."""
    args = _parse_args()

    # Initialize logger
    logging.basicConfig(
        level=str(args.logging).upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("learn2track_create_dataset.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(args)

    # Create dataset from config and save
    with Timer("Generating dataset"):
        _generate_dataset(args.path, args.dir_structure, args.name,
                          args.data_repr, args.streamlines, args.ref,
                          force=args.force)


def _generate_dataset(s_root_path: str, dir_structure: DirStructure, name: str,
                      data_choices: List[str], include_streamlines: bool,
                      streamlines_ref: str = None, force: bool = False):
    """Generate a dataset from a group of dMRI subjects with multiple bundles.
    All bundles are merged as a single whole-brain dataset in voxel space.

    Nomenclature used throughout the code:
    *_path : Path to folder (s_* : string; p_* : Pathlib.Path object)
    *_file : Path to file
    *_image : `Nifti1Image` object
    *_volume : Numpy 3D/4D array

    Parameters
    ----------
    s_root_path : str
        Path to directory root.
    dir_structure : DirStructure
        Directory structure organisation.
    name : str
        Dataset name used to save the .hdf5 file. If not given, guessed from
        path name.
    data_choices : list of str
        Data representation choices.
    include_streamlines : bool
        If true, include streamlines in the dataset.
    streamlines_ref : str
        Data to use as reference when sending streamlines to VOX space.
    force : bool
        Overwrite existing dataset.
    """
    if include_streamlines and streamlines_ref is None:
        raise ValueError("Using --streamlines requires a reference with --ref")

    # Prepare folder/file name
    if name:
        dataset_name = name
    else:
        dataset_name = os.path.basename(s_root_path)

    # Initialize database
    dataset_file = "{}.hdf5".format(dataset_name)

    if Path(dataset_file).exists():
        if force:
            os.remove(dataset_file)
        else:
            raise FileExistsError("Dataset already exists! Use --force "
                                  "to overwrite")

    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 4

        subject_ids = [s.name for s in Path(s_root_path).glob('*/')]

        # Starting subject processing
        logging.info("Processing {} subjects : {}"
                     .format(len(subject_ids), subject_ids))

        for subject_id in subject_ids:
            with Timer("Processing subject {}".format(subject_id)):
                p_subject_path = Path(s_root_path).joinpath(subject_id)

                # Process subject data
                s_subject_path = str(p_subject_path)
                data_reprs = _process_subject_data_reprs(subject_id,
                                                         s_subject_path,
                                                         data_choices,
                                                         dir_structure)
                if data_reprs is None:
                    logging.error("There was an error processing data for "
                                  "subject {}".format(subject_id))
                    continue

                # Add subject to hdf database
                hdf_subject = hdf_file.create_group(subject_id)

                # Add all chosen data representations
                for data_name, data_image in data_reprs.items():
                    hdf_data_group = hdf_subject.create_group(data_name)
                    hdf_data_group.attrs['vox2rasmm'] = data_image.affine
                    hdf_data_group.create_dataset('data',
                                                  data=data_image.get_fdata(
                                                      dtype=np.float32))

                # Add streamlines
                if include_streamlines:
                    subject_output = _process_subject_streamlines(subject_id,
                                                                  s_subject_path,
                                                                  streamlines_ref,
                                                                  dir_structure)
                    streamlines, bundle_to_sid_slice, euclidean_lengths = subject_output

                    if streamlines is not None:
                        streamlines_group = hdf_subject.create_group(
                            'streamlines')
                        # Accessing private Dipy values, but necessary
                        streamlines_group.create_dataset('data',
                                                         data=streamlines._data)
                        streamlines_group.create_dataset('offsets',
                                                         data=streamlines._offsets)
                        streamlines_group.create_dataset('lengths',
                                                         data=streamlines._lengths)
                        streamlines_group.create_dataset('euclidean_lengths',
                                                         data=euclidean_lengths)

                        bundle_group = streamlines_group.create_group(
                            'bundle_to_sid_slice')
                        for bundle_name, slice_data in bundle_to_sid_slice.items():
                            bundle_group.create_dataset(bundle_name,
                                                        data=slice_data,
                                                        dtype=np.int32)
                    else:
                        logging.warning("No streamlines found for subject : {}"
                                        .format(subject_id))
                        # No streamlines found, remove subject data
                        del hdf_subject
                        del hdf_file[subject_id]

    logging.info("Saved dataset : {}".format(dataset_file))


def _process_subject_data_reprs(subject_id: str, s_subject_path: str,
                                data_choices: List[str],
                                dir_structure: DirStructure):
    """Process a subject to extract chosen data representations,
    such as T1, DWI, etc..

    Parameters
    ----------
    subject_id : str
        Subject unique identifier.
    s_subject_path : str
        Path to the subject's data.
    data_choices : list of str
        Data representation choices.
    dir_structure : DirStructure
        Directory structure to fetch from.

    Returns
    -------
    data_reprs : dict of (str, Nifti1Image)
        All data representations as Nifti1Images defined by `data_choices`
    """

    data_reprs = {}
    for data_name in data_choices:
        s_file_path = fetch_data_path(subject_id, s_subject_path, data_name,
                                      dir_structure)
        data_image = nibabel.load(s_file_path)
        data_reprs[data_name] = data_image
    return data_reprs


def _process_subject_streamlines(subject_id: str,
                                 s_subject_path: str,
                                 reference_data_choice: str,
                                 dir_structure: DirStructure):
    """Load and merge a single subject's streamlines from all available
    bundles (if any), while keeping track of a [bundle_id -> streamline_ids]
    map.

    Parameters
    ----------
    subject_id : str
        Subject unique identifier.
    s_subject_path : str
        Path to subject data.
    reference_data_choice : str
        Data to use as reference when sending streamlines to VOX space.
    dir_structure : DirStructure
        The directory structure to use.

    Returns
    -------
    streamlines : ArraySequence
        All streamlines, merged in bundle alphabetical order.
    bundle_to_sid_slice : dict of (str, tuple)
        Dictionary matching a bundle name to a slice of contiguous
        streamline ids.
    euclidean_lengths : list of float
        Euclidean length of each streamline.
    """
    output_tractogram = None
    output_lengths = []
    bundle_to_sid_slice = {}

    s_bundles_paths = fetch_bundles_paths(subject_id, s_subject_path,
                                          dir_structure)
    ref = fetch_data_path(subject_id, s_subject_path, reference_data_choice,
                          dir_structure)

    for bundle_name, s_bundle_path in s_bundles_paths.items():
        tractogram = load_tractogram(s_bundle_path, ref, to_space=Space.RASMM)

        if len(tractogram) == 0:
            logging.error("Tractogram {} contains 0 streamlines, "
                          "skipping...".format(str(s_bundle_path)))
            continue

        # Internal validation check
        tractogram.remove_invalid_streamlines()

        # Compute euclidean lengths
        output_lengths.extend(length(tractogram.streamlines))

        # Add to output tractogram
        if output_tractogram is None:
            output_tractogram = tractogram
        else:
            # Validate that tractograms are in the same space
            assert is_tractogram_in_same_space(output_tractogram, tractogram), \
                "Inconsistent tractogram space: {}".format(tractogram)
            output_tractogram.streamlines.extend(tractogram.streamlines)

        slice_start = len(output_tractogram) - len(tractogram)
        slice_end = len(output_tractogram)
        bundle_to_sid_slice[bundle_name] = (slice_start, slice_end)

        # Finalize processing
        if output_tractogram is not None:
            output_streamlines_rasmm = output_tractogram.streamlines

            # Transfer the streamlines to the reference space before bringing
            # them to VOX space
            # NOTE: This is done in case the streamlines were tracked in a
            # different space than the provided dataset reference
            output_tractogram = StatefulTractogram(output_streamlines_rasmm,
                                                   ref, space=Space.RASMM)

    return output_tractogram, output_lengths, bundle_to_sid_slice


if __name__ == '__main__':
    main()
