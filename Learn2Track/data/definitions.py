"""Definitions for paths and names depending on the directory structure."""
from enum import Enum

_DATA_2_FILENAME_FLAT = {"dwi": "dwi.nii.gz",
                         "dwi_odf": "dwi_sh.nii.gz",
                         "fodf": "fodf.nii.gz",
                         "fodf_peaks": "peaks.nii.gz"}
_DATA_2_FILEPATH_TRACTOFLOW = {
    "dwi": ["Normalize_DWI", "dwi_normalized.nii.gz"],
    "dwi_odf": ["Compute_SH", "dwi_sh.nii.gz"],
    "fodf": ["FODF_Metrics", "fodf.nii.gz"],
    "fodf_peaks": ["FODF_Metrics", "peaks.nii.gz"]}
_BUNDLES_DIR_FLAT = "bundles"
_BUNDLES_DIR_TRACTOFLOW = "Bundles"
_FILE_SPLIT_TOKEN_FLAT = "_"
_FILE_SPLIT_TOKEN_TRACTOFLOW = "__"


class DirStructure(Enum):
    """Supported directory structure
    FLAT structure:
    Root
    |- subid1
       |- subid1_file1.nii.gz
       |- subid1_file2.nii.gz
       ...
       |- bundles
          |- subid_bundle1.trk
          |- subid_bundle2.trk
          ...
    |- subid2
       ...

    TRACTOFLOW structure:
    Root
    |- subid1
       |- Bet_DWI
          |- subid__dwi_bet.nii.gz
          ...
       |- Bundles
          |- subid__bundle1.trk
          |- subid__bundle2.trk
       ...
    |- subid2
       ...
    """
    FLAT = 1
    TRACTOFLOW = 2

    # Needed for argparse using Enum
    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return DirStructure[s]
        except KeyError:
            raise ValueError()
