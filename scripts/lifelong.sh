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
python config_manager.py set_experiment -c $config_name -e $initial_dataset
# Start processes in the background
for ((i=1; i<=generations; i++)); do
    ./scripts/train_active.sh $config_name "generation"$i 
    ./scripts/collect_active.sh $experiment_name"_generation_"$i "generation"$i 
    python config_manager.py set_dataset -c $config_name -d $experiment_name"_generation_"$i
done



echo "All processes have finished."