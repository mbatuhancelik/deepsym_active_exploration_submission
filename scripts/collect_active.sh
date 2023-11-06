#!/usr/bin/zsh

# Number of processes to run
num_data_collectors=5

# Your command to run (replace with your actual command)

# Array to store process PIDs
pids=()
echo "collecting dataset $1 "
# Start processes in the background
mkdir ./data/$1
for ((i=0; i<num_data_collectors; i++)); do
  nohup python explore_council.py -N 1000 -T 10 -i $i -o ./data/$1 -d cuda:0 &
  pids+=($!)  # Store the PID of the background process
done

for ((i=0; i<num_data_collectors; i++)); do
  nohup python explore_council.py -N 1000 -T 10 -i $((i + num_data_collectors)) -o ./data/$1 -d cuda:1&
  pids+=($!)  # Store the PID of the background process
done

# Wait for all processes to finish
for pid in "${pids[@]}"; do
  wait $pid
done


i=$((num_data_collectors * 2))
python dataset_utils.py merge_rolls -o $1 -i $i

echo "All processes have finished."
