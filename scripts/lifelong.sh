#!/usr/bin/zsh

# Number of processes to run
num_trainers=3
experiment_name=$1
initial_dataset=$2
generations=$3

config_name=$experiment_name"_config.yaml"
# Array to store process PIDs
pids=()
mkdir council
cp config.yaml $config_name
python config_manager.py set_generation -g 0 -c $config_name
python config_manager.py set_dataset -c $config_name -d $initial_dataset
python config_manager.py set_experiment -c $config_name -e $experiment_name
# Start processes in the background
for ((i=0; i<=generations; i++)); do
    python config_manager.py set_generation -g $i -c $config_name
    ./scripts/train_active.sh $config_name "generation"$i 
    ./scripts/collect_active.sh $experiment_name"_collection_"$i "generation"$i 
    python dataset_utils.py merge_datasets -o $experiment_name"_generation_"$i -l $experiment_name"_collection_"$i -s $initial_dataset
    python config_manager.py set_dataset -c $config_name -d $experiment_name"_generation_"$i
    initial_dataset=$experiment_name"_generation_"$i
    python config_manager.py set_generation -g $i -c $config_name
done
# ./scripts/train_active.sh $config_name "generation"$((i+1))


echo "All processes have finished."