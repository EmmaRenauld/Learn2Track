#!/bin/bash


###############
# Choose working place
###############
working_folder=$1


###############
# Load variables
###############
echo "Declaring bash variables for databases ismrm2015, neher99, hcp, tractoinferno"

######
# For ismrm2015:
#     mkdir $ismrm2015_folder/original
#     mkdir $ismrm2015_folder/preprocessed
######
    ismrm2015_folder="$working_folder/phantoms_or_simulated/ismrm2015/"

    database_folder_ismrm2015_noArtefact="$ismrm2015_folder/preprocessed/noArtefact"
    subject_list_ismrm2015_noArtefact="$ismrm2015_folder/derivatives/noArtefact/subjects.txt"

    database_folder_ismrm2015_basic="$ismrm2015_folder/derivatives/basic"
    subject_list_ismrm2015_basic="$ismrm2015_folder/derivatives/basic/subjects.txt"

#####
# For neher99:
#####
    neher99_folder="$working_folder/phantoms_or_simulated/neher99/"

    database_folder_neher99_noArtefact="$neher99_folder/derivatives/noArtefact"
    subject_list_neher99_noArtefact="$neher99_folder/derivatives/noArtefact/subjects.txt"

    database_folder_neher99="$neher99_folder/derivatives/basic"
    subject_list_neher99="$neher99_folder/derivatives/basic/subjects.txt"

#####
# For HCP:
#     mkdir $hcp_folder/original  (= input tractoflow)
#         -> see on Beluga: /home/renaulde/projects/rrg-descotea/datasets/hcp_1200/derivatives/tractoflow/input
#     mkdir $hcp_folder/preprocessed (= output tractoflow + recobundles)
#     cd $hcp_folder/preprocessed
#         cp -as /home/renaulde/projects/rrg-descotea/datasets/hcp_1200/derivatives/tractoflow/output/results_ranByEmma/results/ ./
#####
    hcp_folder="$working_folder/in_vivo_database/hcp_1200/derivatives"

    database_folder_hcp=$hcp_folder
    subject_list_hcp="$hcp_folder/subjects_for_ML.txt"

#####
# For TractoInferno:
#####
    tractoinferno_folder="$working_folder/in_vivo_database/tractoinferno/"

    database_folder_tractoinferno="$tractoinferno_folder/derivatives/"
    subject_list_tractoinferno="$tractoinferno_folder/derivatives/subjects.txt"

    database_folder_test_retest=$database_folder_tractoinferno
    subject_list_test_retest="$tractoinferno_folder/derivatives/subjects_test_retest.txt"

