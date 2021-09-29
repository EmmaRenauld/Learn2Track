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
    ismrm2015_folder="$working_folder/phantoms_or_simulated/ismrm2015/derivatives"

    database_folder_ismrm2015_noArtefact="$ismrm2015_folder/noArtefact"
    subjects_list_ismrm2015_noArtefact="$ismrm2015_folder/noArtefact/subjects.txt"
    training_subjs_ismrm2015_noArtefact="$ismrm2015_folder/noArtefact/ML_studies/subjects_for_ML_training.txt"
    validation_subjs_ismrm2015_noArtefact="$ismrm2015_folder/noArtefact/ML_studies/subjects_for_ML_validation.txt"
    config_file_ismrm2015_noArtefact="$ismrm2015_folder/noArtefact/ML_studies/config_file.json"

    database_folder_ismrm2015_basic="$ismrm2015_folder/derivatives/basic"
    subjects_list_ismrm2015_basic="$ismrm2015_folder/derivatives/basic/subjects.txt"

#####
# For neher99:
#     original:
#         -> see on Beluga: /home/renaulde/projects/rrg-descotea/datasets/hcp_simulated/
#     preprocessed: see all_steps, tractoflow was ran
#         cp -n -r -v ~/scratch/neher99/results/* preprocessed  # the n option is to avoid overwriting what has already been transfered
#####
    neher99_folder="$working_folder/phantoms_or_simulated/neher99/derivatives"

    database_folder_neher99_noArtefact="$neher99_folder/noArtefact"
    subjects_list_neher99_noArtefact="$neher99_folder/noArtefact/subjects.txt"

    database_folder_neher99_basic="$neher99_folder/basic"
    subjects_list_neher99_basic="$neher99_folder/basic/subjects.txt"
    training_subjs_neher99_basic="$neher99_folder/basic/ML_results/subjects_for_ML_training.txt"
    validation_subjs_neher99_basic="$hcp_folder/basic/ML_results/subjects_for_ML_validation.txt"
    testing_subjs_neher99_basic="$hcp_folder/basic/ML_results/subjects_for_ML_testing.txt"
    config_file_neher99_basic="$hcp_folder/basic/ML_results/config_file_hcp.json"

#####
# For HCP:
#     original:
#         mkdir $hcp_folder/original  (= input tractoflow)
#         -> see on Beluga: /home/renaulde/projects/rrg-descotea/datasets/hcp_1200/derivatives/tractoflow/input
#     preprocessed:  I ran it some time ago and put it on beluga, so:
#         mkdir $hcp_folder/preprocessed (= output tractoflow + recobundles)
#         cd $hcp_folder/preprocessed
#         cp -as /home/renaulde/projects/rrg-descotea/datasets/hcp_1200/derivatives/tractoflow/output/results_ranByEmma/results/ ./
#     subjs:
#         echo sub-* > subjects.txt
#     config file:
#####
    hcp_folder="$working_folder/in_vivo_databases/hcp_1200/derivatives"

    database_folder_hcp=$hcp_folder
    subjects_list_hcp="$hcp_folder/subjects_for_ML.txt"
    training_subjs_hcp="$hcp_folder/ML_results/subjects_for_ML_training.txt"
    validation_subjs_hcp="$hcp_folder/ML_results/subjects_for_ML_validation.txt"
    testing_subjs_hcp="$hcp_folder/ML_results/subjects_for_ML_testing.txt"
    config_file_hcp="$hcp_folder/ML_results/config_file_hcp.json"

#####
# For TractoInferno:
#####
    tractoinferno_folder="$working_folder/in_vivo_databases/tractoinferno/"

    database_folder_tractoinferno="$tractoinferno_folder/derivatives/"
    subjects_list_tractoinferno="$tractoinferno_folder/derivatives/subjects.txt"

    database_folder_test_retest=$database_folder_tractoinferno
    subjects_list_test_retest="$tractoinferno_folder/derivatives/subjects_test_retest.txt"

