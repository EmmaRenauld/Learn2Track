
# Run this from inside Learn2track

###########
# Organize data
###########
beluga="YOUR INFOS"
ismrm2015_folder="$beluga/ismrm2015/derivatives"
# or
ismrm2015_folder=~/Documents/phantoms_or_simulated/ismrm2015/derivatives/

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

# Note Emma: my /data is not on the same disk. Put the data on / (ex, home or
# documents) and run the experiment in /data.
experiment_folder='/data/rene2201/experiments'
learn2track='/home/local/USHERBROOKE/rene2201/my_applications/scil_vital/Learn2Track'
cd $experiment_folder
mkdir $experiment_folder/Learn2track
rm -r $experiment_folder/Learn2track/test_experiment1/checkpoint
# If bugged before:
# rm -r $experiment_folder/Learn2track/test_experiment1/model_old

# run this with a max_epoch=1 to test.
python $learn2track/scripts/train_model.py --loggin info \
      --input_group 'input' --target_group 'streamlines' \
      --hdf5_file $database_folder/hdf5/ismrm2015_noArtefact_test.hdf5 \
      --parameters_filename $learn2track/parameters/training_parameters_experimentX.yaml \
      --experiment_name test_experiment1 $experiment_folder/Learn2track

# Then, try again, it will load from checkpoint!
python $learn2track/scripts/train_model.py --loggin info \
      --input_group 'input' --target_group 'streamlines' \
      --hdf5_file $database_folder/hdf5/ismrm2015_noArtefact_test.hdf5 \
      --parameters_filename $learn2track/parameters/training_parameters_experimentX.yaml \
      --experiment_name test_experiment1 $experiment_folder/Learn2track \
      --override_checkpoint_patience 3 --override_checkpoint_max_epochs 3
