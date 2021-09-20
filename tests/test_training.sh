
# Run this from inside Learn2track

###########
# Organize data
###########
beluga="YOUR INFOS"
ismrm2015_folder="$beluga/ismrm2015/derivatives"

database_folder="$ismrm2015_folder/noArtefact"
name=ismrm2015_noArtefact_test
hdf5_filename="$database_folder/hdf5/$name.hdf5"

################
# 1. See running_tests in dwi_ml for tests on the first
# parts.
# - organize_from_tractoflow
# - create hdf5
################

###########
# 2. Running training on chosen database:
###########
mkdir $database_folder/experiments
rm -r $database_folder/experiments/test_experiment1/checkpoint
# If bugged before:
rm -r $database_folder/experiments/test_experiment1/model_old/model
python scripts/train_model.py --loggin info \
      --input_group 'input' --target_group 'streamlines' \
      --hdf5_filename $database_folder/hdf5/ismrm2015_noArtefact_test.hdf5 \
      --parameters_filename parameters/training_parameters_experimentX.yaml \
      --experiment_name test_experiment1 $database_folder/experiments
