#!/usr/bin/zsh

# Number of processes to run
num_data_collectors=7

# Your command to run (replace with your actual command)
N=600
# Array to store process PIDs
pids=()
echo "collecting dataset $1 "
# Start processes in the background
# mkdir ./data/$1
for ((i=0; i<num_data_collectors; i++)); do
  if [ ! -e "./data/$1/state_$i.pt" ]; 
  then
  echo "File '$i' created."
  CUDA_VISIBLE_DEVICES=0 nohup python explore_council.py -N $N -T 10 -i $i -o ./data/$1  -gen $3 &
  pids+=($!)  # Store the PID of the background process
  fi
done

for ((i=0; i<num_data_collectors; i++)); do
  if [ ! -e "./data/$1/state_$i.pt" ]; 
  then
  CUDA_VISIBLE_DEVICES=1 nohup python explore_council.py -N $N -T 10 -i $((i + num_data_collectors)) -o ./data/$1 -gen $3 &
  pids+=($!)  # Store the PID of the background process
  fi
done

# Wait for all processes to finish
for pid in "${pids[@]}"; do
  wait $pid
done


i=$((num_data_collectors * 2))
python dataset_utils.py merge_rolls -o $1 -i $i

echo "$1 collected"
