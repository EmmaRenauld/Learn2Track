# Step A = preparation
# Step B = Running. Similar to dwi_ml/please_copy_and_adapt/ALL_STEPS.sh



#########
# A.0. Choose study
#########

# To run on my computer:
my_bash_scripts="/home/local/USHERBROOKE/rene2201/my_applications/scil_vital/learn2track/USER_SCRIPTS/"
data_root="/home/local/USHERBROOKE/rene2201/Documents"

# To run on Beluga:
my_bash_scripts="/home/renaulde/my_applications/learn2track/USER_SCRIPTS/"
data_root="/home/renaulde/projects/rrg-descotea/renaulde"

study=fibercup
study=ismrm2015_noArtefact
study=ismrm2015_basic
study=neher99_noArtefact
study=neher99_basic
study=hcp
study=tractoinferno
study=test_retest

#########
# Find folders and subject lists
#########

# Load "global" variables: the names of my folders
    source $my_bash_scripts/my_variables.sh $data_root
    eval database_folder=\${database_folder_$study}
    eval subjects_list=\${subjects_list_$study}
    cd $database_folder


# Printing infos on current study
    echo -e "=========LEARN2TRACK\n" \
         "     Chosen study: $study \n"         \
         "     Input data: $database_folder \n" \
         "     Subject list: $subjects_list \n"  \
         "     Please verify that tree contains original (ex, tractoflow input) + preprocessed (ex, tractoflow output)"
    tree -d -L 2 $database_folder
    cat $subjects_list


#########
# A.1. Organize from tractoflow
#########
    rm -r $database_folder/dwi_ml_ready
    bash $my_bash_scripts/organize_from_tractoflow.sh $database_folder $subjects_list
    tree -d -L 2 $database_folder
    first_subj=`ls $database_folder/dwi_ml_ready | awk -F' ' '{ print $1}'`
    tree $database_folder/dwi_ml_ready/$first_subj


#########
# A.2. Organize from recobundles
#########
    bash $my_bash_scripts/organize_from_recobundles.sh $database_folder RecobundlesX/multi_bundles $subjects_list


# ===========================================================================

#########
# B.1. Create hdf5 dataset
#########
    # Choosing the parameters for this study
    eval config_file=\${config_file_$study}
    eval training_subjs=\${training_subjs_$study}
    eval validation_subjs=\${validation_subjs_$study}
    now=`date +'%Y_%d_%m_%HH%MM'`
    name=${study}_$now

    # Choosing bundles
    #eval bundles_txt=\${bundles_$study}
    #bundles=$(<bundles.txt)
    #option_bundles="--bundles $bundles"
    option_bundles=""

    #option_logging="--logging debug"
    option_logging="--logging info"
    option_logging=""

    # Paramaters that I keep fixed for all studies
    mask_for_standardization="masks/wm_mask.nii.gz"
    space="rasmm"  # {rasmm,vox,voxmm}

    echo -e "=========RUNNING LEARN2TRACK HDF5 DATASET CREATION\n" \
         "     Chosen study: $study \n"         \
         "     Config file: $config_file \n"       \
         "     Training subjects: $training_subjs \n"  \
         "     Validation subjects: $validation_subjs \n"  \
         "     Bundles: $option_bundles"

    # Preparing hdf5.
    create_hdf5_dataset.py --force --name $name --std_mask $mask_for_standardization \
        $option_bundles $option_logging --space $space $database_folder $config_file \
        $training_subjs $validation_subjs --enforce_bundles_presence True

############
# B.2. Train model
############

    python $learn2track/scripts_python/train_model.py --loggin info \
        --input_group 'input' --target_group 'streamlines' \
        --hdf5_file $database_folder/hdf5/ismrm2015_noArtefact_test.hdf5 \
        --yaml_parameters $learn2track/parameters/training_parameters_experimentX.yaml \
        --experiment_name test_experiment1 $experiment_folder/Learn2track
