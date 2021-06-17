# To run on my computer:
my_bash_scripts="/home/local/USHERBROOKE/rene2201/my_applications/scil_vital/learn2track/USER_SCRIPTS/"
data_root="~/Documents"

# To run on Beluga:
my_bash_scripts="/home/renaulde/my_applications/learn2track/USER_SCRIPTS/"
data_root="/home/renaulde/projects/rrg-descotea/renaulde"

#########
# 0. Choose study
#########
#study=ismrm2015_noArtefact
#study=ismrm2015_basic
#study=neher99
study=hcp
#study=tractoinferno
#study=test_retest

#########
# 1. Find folders and subject lists
#########
source $my_bash_scripts/1_my_variables.sh $data_root

# Choosing the ones we want for this study
eval database_folder=\${database_folder_$study}
eval subject_list=\${subject_list_$study}

echo -e "RUNNING LEARN2TRACK PREPARATION\n" \
     "     Chosen study: $study \n"         \
     "     Input data: $database_folder \n" \
     "     Subject list: $subject_list \n"  \
     "     Please verify that tree contains original (ex, tractoflow input) + preprocessed (ex, tractoflow output)"
cd $database_folder
tree -d -L 2 $database_folder

cat $subject_list

#########
# 2. Organize from tractoflow
#########
bash $my_bash_scripts/2_organize_from_tractoflow.sh $database_folder $subject_list
tree -d -L 2 $database_folder

#########
# 3. Organize from recobundles
#########
bash $my_bash_scripts/3_organize_from_recobundles.sh $database_folder RecobundlesX/multi_bundles $subject_list
tree -d -L 2 $database_folder
#HCP: tree $database_folder/dwi_ml_ready/sub-100206
