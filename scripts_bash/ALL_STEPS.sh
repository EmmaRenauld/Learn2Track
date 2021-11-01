
#########
# Choose study
#########
study=MY_STUDY
database_folder=MY_PATH
subjects_list=SUBJS.txt

#########
# Find folders and subject lists
#########

# Printing infos on current study
    echo -e "=========LEARN2TRACK\n" \
         "     Chosen study: $study \n"         \
         "     Input data: $database_folder \n" \
         "     Subject list: $subjects_list \n"  \
         "     Please verify that tree contains original (ex, tractoflow input) + preprocessed (ex, tractoflow output)"
    tree -d -L 2 $database_folder
    cat $subjects_list


#########
# Organize from tractoflow
#########
    rm -r $database_folder/dwi_ml_ready
    bash $my_bash_scripts/organize_from_tractoflow.sh $database_folder $subjects_list
    tree -d -L 2 $database_folder
    first_subj=`ls $database_folder/dwi_ml_ready | awk -F' ' '{ print $1}'`
    tree $database_folder/dwi_ml_ready/$first_subj


#########
# Organize from recobundles
#########
    bash $my_bash_scripts/organize_from_recobundles.sh $database_folder RecobundlesX/multi_bundles $subjects_list


# ===========================================================================

#########
# Create hdf5 dataset
#########
    # Choosing the parameters for this study
    eval config_file=\${config_file_$study}
    eval training_subjs=\${training_subjs_$study}
    eval validation_subjs=\${validation_subjs_$study}
    now=`date +'%Y_%d_%m_%HH%MM'`
    name=${study}_$now

    # Paramaters that I keep fixed for all studies
    mask="masks/wm_mask.nii.gz"
    space="rasmm"  # {rasmm,vox,voxmm}

    echo -e "=========RUNNING LEARN2TRACK HDF5 DATASET CREATION\n" \
         "     Study: $name \n" \
         "     Config file: $config_file \n"       \
         "     Training subjects: $training_subjs \n"  \
         "     Validation subjects: $validation_subjs \n" \
         "     mask for standardization: $mask \n" \
         "     Complete config_file infos: \n"
    cat $config_file

    # Preparing hdf5.
    create_hdf5_dataset.py --force --name $name --std_mask $mask \
        --logging info --space $space --enforce_files_presence True \
        --independent_modalities True \
        $database_folder/dwi_ml_ready $database_folder $config_file \
        $training_subjs $validation_subjs

############
# Train model
############

    python train_model.py --loggin info \
        --input_group 'input' --target_group 'streamlines' \
        --hdf5_file $database_folder/hdf5/ismrm2015_noArtefact_test.hdf5 \
        --yaml_parameters $learn2track/parameters/training_parameters_experimentX.yaml \
        --experiment_name test_experiment1 $experiment_folder/Learn2track

    # Re-run from checkpoint
    python train_model.py --loggin info \
        --experiment_name test_experiment1 --experiment_path $main_folder/experiments/Learn2Track \
        --override_checkpoint_max_epochs 20